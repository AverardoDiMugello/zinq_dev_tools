use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::CString;
use std::io::Read;
use std::path::PathBuf;

use petgraph::graph::NodeIndex;
use petgraph::visit::Dfs;
use petgraph::Direction;

/// Re-export of selected isla-lib exports
mod isla {
    pub use isla_lib::bitvector::b64::B64;
    pub use isla_lib::bitvector::BV;
    pub use isla_lib::config::{ISAConfig, Tool};
    pub use isla_lib::init::{initialize_architecture, Initialized};
    pub use isla_lib::ir::ssa::{BlockInstr, BlockLoc, Edge, SSAName, Terminator, CFG};
    pub use isla_lib::ir::{
        label_instrs, prune_labels, AssertionMode, ExitCause, Exp, IRTypeInfo, Instr, Name, Op,
        SharedState, Symtab, Ty, UVal, Val, ABSTRACT_PRIMOP, CURRENT_EXCEPTION, HAVE_EXCEPTION,
        INSTR_ANNOUNCE, INTERRUPT_PENDING, READ_REGISTER_FROM_VECTOR, REG_DEREF, RESET_REGISTERS,
        RETURN, SAIL_EXCEPTION, THROW_LOCATION, WRITE_REGISTER_FROM_VECTOR,
    };
    pub use isla_lib::ir_lexer::new_ir_lexer;
    pub use isla_lib::ir_parser::IrParser;
    pub use isla_lib::primop::{Binary, Primops, Unary, Variadic};
    pub use isla_lib::register::RelaxedVal;
    pub use isla_lib::zencode;
}
use isla::BV;

/// Re-export of selected inkwell exports
mod llvm {
    pub use inkwell::attributes::{Attribute, AttributeLoc};
    pub use inkwell::basic_block::BasicBlock;
    pub use inkwell::builder::Builder;
    pub use inkwell::context::Context;
    pub use inkwell::module::{Linkage, Module};
    pub use inkwell::types::{
        AnyType, AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType,
        IntType, StructType,
    };
    pub use inkwell::values::{
        ArrayValue, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, GlobalValue,
        IntValue, PointerValue, StructValue,
    };
    pub use inkwell::{
        AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel, ThreadLocalMode,
    };
}
use llvm::{BasicType, BasicValue};

mod externs;

struct J2LStaticContext<'a> {
    registers: HashMap<isla::Name, llvm::GlobalValue<'a>>,
    lets: HashMap<isla::Name, llvm::GlobalValue<'a>>,
    /// Jib Name to the corresponding llvm FunctionValue and the types its params point to
    functions: HashMap<
        isla::Name,
        (
            llvm::FunctionValue<'a>,
            Vec<llvm::BasicMetadataTypeEnum<'a>>,
        ),
    >,
    primops: J2LPrimops<'a>,
    types: J2LDeclaredTypes<'a>,
    symtab: isla::Symtab<'a>,
}

impl<'a> J2LStaticContext<'a> {
    fn new(
        jib_spec: &isla::Initialized<'a, isla::B64>,
        module: &llvm::Module<'a>,
        llvm_ctx: &'a llvm::Context,
    ) -> Self {
        let types = J2LDeclaredTypes::new(&jib_spec.shared_state, module, llvm_ctx);

        let mut registers = HashMap::new();
        for (id, ty) in &jib_spec.shared_state.registers {
            let name = format!(
                "{}",
                isla::zencode::decode(jib_spec.shared_state.symtab.to_str_demangled(*id))
            );
            let glob = j2l_register_decl(&name, ty.clone(), &types, module, llvm_ctx);
            assert!(registers.insert(*id, glob).is_none());
        }

        // Reg bindings are the (potential) initializers for each register
        for (id, reg) in jib_spec.regs.iter() {
            if let isla::RelaxedVal::Init { last_write, .. } = &reg.value {
                let init = j2l_static_val(last_write, &types, &registers, llvm_ctx);
                let glob_var = registers.get(id).unwrap();
                glob_var.set_initializer(&init);
            }
        }

        // Let bindings are constants that are statically initialized with some instructions
        let mut lets = HashMap::new();
        for (id, val) in jib_spec.lets.iter() {
            let name = jib_spec.shared_state.symtab.to_str(id.clone());
            match val {
                isla::UVal::Init(val) => {
                    let init = j2l_static_val(val, &types, &registers, llvm_ctx);
                    let glob_var = module.add_global(
                        init.get_type(),
                        Some(llvm::AddressSpace::default()),
                        name,
                    );

                    glob_var.set_constant(true);
                    glob_var.set_initializer(&init);

                    let prev = lets.insert(id.clone(), glob_var);
                    assert_eq!(prev, None);
                }
                isla::UVal::Uninit(ty) => panic!("Uninit {name}: {ty:?}"),
            }
        }

        // Add the global extern pointer to the Zinq memory map
        let zinq_mem_map = module.add_global(
            llvm_ctx.i8_type(),
            Some(llvm::AddressSpace::default()),
            "zinq_mem_map",
        );
        // Add the two builtin Zinq functions for memory read/write
        let zinq_read_mem_ty = llvm_ctx.i64_type().fn_type(
            &[
                llvm_ctx.ptr_type(llvm::AddressSpace::default()).into(), // Zinq mem map
                llvm_ctx.i64_type().into(),
                llvm_ctx.i64_type().into(),
            ],
            false,
        );
        let zinq_read_mem = module.add_function("zinq_read_mem", zinq_read_mem_ty, None);

        let zinq_write_mem_ty = llvm_ctx.void_type().fn_type(
            &[
                llvm_ctx.ptr_type(llvm::AddressSpace::default()).into(), // Zinq mem map
                llvm_ctx.i64_type().into(),
                llvm_ctx.i64_type().into(),
                llvm_ctx.i64_type().into(),
            ],
            false,
        );
        let zinq_write_mem = module.add_function("zinq_write_mem", zinq_write_mem_ty, None);

        let mut functions = HashMap::new();
        let mut primops = J2LPrimops::new();
        let isla_primops = isla::Primops::default();
        for (id, (param_tys, ret_ty, ext_name)) in &jib_spec.shared_state.externs {
            let fn_name = format!(
                "{}",
                isla::zencode::decode(jib_spec.shared_state.symtab.to_str_demangled(*id))
            );
            let ret_ty = j2l_ty((*ret_ty).clone(), &types, llvm_ctx);
            let mut param_pointee_tys = param_tys
                .iter()
                .map(|ty| j2l_ty(ty.clone(), &types, llvm_ctx))
                .map(|ty| llvm::BasicMetadataTypeEnum::try_from(ty).unwrap())
                .collect::<Vec<llvm::BasicMetadataTypeEnum>>();
            param_pointee_tys.insert(0, llvm::BasicMetadataTypeEnum::try_from(ret_ty).unwrap());

            let mut param_tys = Vec::with_capacity(param_pointee_tys.len());
            for _ in 0..param_pointee_tys.len() {
                param_tys.push(llvm::BasicMetadataTypeEnum::from(
                    llvm_ctx.ptr_type(llvm::AddressSpace::default()),
                ));
            }

            let ty = llvm_ctx.void_type().fn_type(&param_tys, false);
            let f = module.add_function(&fn_name, ty, None);
            llvm_set_fn_attrs(&f, &param_pointee_tys, llvm_ctx);
            externs::fill_if_mem_access(
                &ext_name,
                f,
                &types,
                &llvm_ctx,
                zinq_mem_map,
                zinq_read_mem,
                zinq_write_mem,
            );
            externs::fill_if_supported(&ext_name, f, &param_pointee_tys, &types, &llvm_ctx);

            // Find the isla::Primop fn value for this extern (if it is one) and insert it into J2LPrimops
            if let Some(unop) = isla_primops.unary.get(*ext_name) {
                let primop_variations = primops
                    .unary
                    .entry(unop.clone())
                    .or_insert_with(|| HashMap::new());
                primop_variations.insert(
                    absurd_hash_of_function_params(&param_pointee_tys),
                    (ext_name.to_string(), f, param_pointee_tys),
                );
            } else if let Some(binop) = isla_primops.binary.get(*ext_name) {
                let primop_variations = primops
                    .binary
                    .entry(binop.clone())
                    .or_insert_with(|| HashMap::new());
                primop_variations.insert(
                    absurd_hash_of_function_params(&param_pointee_tys),
                    (ext_name.to_string(), f, param_pointee_tys),
                );
            } else if let Some(varop) = isla_primops.variadic.get(*ext_name) {
                let primop_variations = primops
                    .variadic
                    .entry(varop.clone())
                    .or_insert_with(|| HashMap::new());
                primop_variations.insert(
                    absurd_hash_of_function_params(&param_pointee_tys),
                    (ext_name.to_string(), f, param_pointee_tys),
                );
            }
            // Isla also has some builtin functions that are treated as externs but NOT turned into primops
            else if *ext_name == "reg_deref" {
                let prev = functions.insert(isla::REG_DEREF, (f, param_pointee_tys));
                assert_eq!(prev, None);
            } else if *ext_name == "reset_registers" {
                let prev = functions.insert(isla::RESET_REGISTERS, (f, param_pointee_tys));
                assert_eq!(prev, None);
            } else if *ext_name == "interrupt_pending" {
                let prev = functions.insert(isla::INTERRUPT_PENDING, (f, param_pointee_tys));
                assert_eq!(prev, None);
            } else if *ext_name == "read_register_from_vector" {
                let prev =
                    functions.insert(isla::READ_REGISTER_FROM_VECTOR, (f, param_pointee_tys));
                assert_eq!(prev, None);
            } else if *ext_name == "write_register_from_vector" {
                let prev =
                    functions.insert(isla::WRITE_REGISTER_FROM_VECTOR, (f, param_pointee_tys));
                assert_eq!(prev, None);
            } else if *ext_name == "instr_announce" || *ext_name == "platform_instr_announce" {
                let prev = functions.insert(isla::INSTR_ANNOUNCE, (f, param_pointee_tys));
                assert_eq!(prev, None);
            } else {
                let prev = functions.insert(id.clone(), (f, param_pointee_tys));
                assert_eq!(prev, None);
            }
        }

        for (id, (params, ret_ty, _)) in &jib_spec.shared_state.functions {
            let name = format!(
                "{}",
                isla::zencode::decode(jib_spec.shared_state.symtab.to_str_demangled(*id))
            );

            let ret_ty = j2l_ty((*ret_ty).clone(), &types, llvm_ctx);
            let mut param_pointee_tys = params
                .iter()
                .map(|(_, param_ty)| j2l_ty((*param_ty).clone(), &types, llvm_ctx))
                .map(|ty| llvm::BasicMetadataTypeEnum::try_from(ty).unwrap())
                .collect::<Vec<llvm::BasicMetadataTypeEnum>>();
            param_pointee_tys.insert(0, llvm::BasicMetadataTypeEnum::try_from(ret_ty).unwrap());

            let mut param_tys = Vec::with_capacity(param_pointee_tys.len());
            for _ in 0..param_pointee_tys.len() {
                param_tys.push(llvm::BasicMetadataTypeEnum::from(
                    llvm_ctx.ptr_type(llvm::AddressSpace::default()),
                ));
            }

            let ty = llvm_ctx.void_type().fn_type(&param_tys, false);
            let f = module.add_function(&name, ty, None);
            llvm_set_fn_attrs(&f, &param_pointee_tys, llvm_ctx);

            assert!(functions.insert(*id, (f, param_pointee_tys)).is_none());
        }

        add_sail_builtins(
            &mut registers,
            &mut primops,
            &isla_primops,
            &types,
            module,
            llvm_ctx,
        );

        // for (_, variations) in primops.unary.iter() {
        //     if variations.len() > 1 {
        //         for (arg_tys, (name, ..)) in variations.iter() {
        //             println!("{name}: {arg_tys}");
        //         }
        //     }
        // }

        // for (_, variations) in primops.binary.iter() {
        //     if variations.len() > 1 {
        //         for (arg_tys, (name, ..)) in variations.iter() {
        //             println!("{name}: {arg_tys}");
        //         }
        //     }
        // }

        // for (_, variations) in primops.variadic.iter() {
        //     if variations.len() > 1 {
        //         for (arg_tys, (name, ..)) in variations.iter() {
        //             println!("{name}: {arg_tys}");
        //         }
        //     }
        // }
        // panic!("");

        J2LStaticContext {
            registers,
            lets,
            functions,
            primops,
            types,
            symtab: jib_spec.shared_state.symtab.clone(),
        }
    }

    fn name(&self, id: &isla::Name) -> String {
        self.symtab.to_str(id.clone()).to_string()
    }

    fn ssa_name(&self, id: &isla::SSAName) -> String {
        format!(
            "{0}.{1}",
            self.symtab.to_str(id.clone().base_name()),
            id.clone().ssa_number()
        )
    }

    fn get_fn(
        &self,
        f: &isla::Name,
    ) -> Option<(llvm::FunctionValue<'a>, &Vec<llvm::BasicMetadataTypeEnum>)> {
        self.functions
            .get(f)
            .and_then(|(f, param_pointee_tys)| Some((f.clone(), param_pointee_tys)))
    }

    fn get_tag(&self, ctor: &isla::Name) -> (llvm::IntValue<'a>, llvm::BasicTypeEnum<'a>) {
        self.types
            .union_ctors
            .get(ctor)
            .and_then(|(_, glob, ty)| {
                Some((glob.get_initializer().unwrap().into_int_value(), ty.clone()))
            })
            .unwrap()
    }
}

fn add_sail_builtins<'a>(
    registers: &mut HashMap<isla::Name, llvm::GlobalValue<'a>>,
    primops: &mut J2LPrimops<'a>,
    isla_primops: &isla::Primops<isla::B64>,
    types: &J2LDeclaredTypes<'a>,
    module: &llvm::Module<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    // Sail has some built-in constants for exception handling that are treated as global registers
    let exception_ty = types
        .mappings
        .get(&isla::SAIL_EXCEPTION)
        .and_then(|ty| {
            if let J2LType::Union(s) = ty {
                Some(s)
            } else {
                None
            }
        })
        .unwrap();
    let current_exception = module.add_global(
        exception_ty.clone(),
        Some(llvm::AddressSpace::default()),
        "CURRENT_EXCEPTION",
    );
    let prev = registers.insert(isla::CURRENT_EXCEPTION, current_exception);
    assert_eq!(prev, None);

    let have_exception = module.add_global(
        llvm_ctx.bool_type(),
        Some(llvm::AddressSpace::default()),
        "HAVE_EXCEPTION",
    );
    let prev = registers.insert(isla::HAVE_EXCEPTION, have_exception);
    assert_eq!(prev, None);

    let throw_location = module.add_global(
        // Empty string. This will never be modified because array assignments to globals are converted to no-ops
        llvm_ctx.i8_type().array_type(0),
        Some(llvm::AddressSpace::default()),
        "THROW_LOCATION",
    );
    let prev = registers.insert(isla::THROW_LOCATION, throw_location);
    assert_eq!(prev, None);

    // Isla has some built-in functions that are treated as externs and turned into primops
    // zupdate_fbits

    let param_pointee_tys = vec![
        llvm::BasicMetadataTypeEnum::from(types.bitvec.clone()), // (unit) RETURN
        llvm::BasicMetadataTypeEnum::from(types.bitvec.clone()),
        llvm::BasicMetadataTypeEnum::from(llvm_ctx.i64_type()),
        llvm::BasicMetadataTypeEnum::from(types.bitvec.clone()),
    ];
    let param_ty =
        llvm::BasicMetadataTypeEnum::from(llvm_ctx.ptr_type(llvm::AddressSpace::default()));
    let param_tys = vec![
        param_ty.clone(),
        param_ty.clone(),
        param_ty.clone(),
        param_ty,
    ];
    let fn_ty = llvm_ctx.void_type().fn_type(&param_tys, false);
    let bitvector_update =
        // module.add_function("zupdate_fbits", fn_ty, Some(llvm::Linkage::External));
        module.add_function("zupdate_fbits", fn_ty, None);
    llvm_set_fn_attrs(&bitvector_update, &param_pointee_tys, llvm_ctx);
    externs::fill_if_supported(
        "zupdate_fbits",
        bitvector_update,
        &param_pointee_tys,
        &types,
        &llvm_ctx,
    );

    let primop_name = String::from("bitvector_update");
    let primop = *isla_primops.variadic.get(&primop_name).unwrap();

    let primop_variations = primops
        .variadic
        .entry(primop)
        .or_insert_with(|| HashMap::new());
    primop_variations.insert(
        absurd_hash_of_function_params(&param_pointee_tys),
        (primop_name, bitvector_update, param_pointee_tys),
    );

    // zsail_assert
    let param_pointee_tys = vec![
        llvm::BasicMetadataTypeEnum::from(types.bitvec.clone()), // (unit) RETURN
        llvm::BasicMetadataTypeEnum::from(llvm_ctx.bool_type()),
        llvm::BasicMetadataTypeEnum::from(llvm_ctx.i8_type().array_type(0)),
    ];
    let param_ty =
        llvm::BasicMetadataTypeEnum::from(llvm_ctx.ptr_type(llvm::AddressSpace::default()));
    let param_tys = vec![param_ty.clone(), param_ty.clone(), param_ty];
    let fn_ty = llvm_ctx.void_type().fn_type(&param_tys, false);
    let sail_assert = module.add_function("zsail_assert", fn_ty, None);
    llvm_set_fn_attrs(&sail_assert, &param_pointee_tys, llvm_ctx);
    externs::fill_if_supported(
        "zsail_assert",
        sail_assert,
        &param_pointee_tys,
        &types,
        &llvm_ctx,
    );

    let primop_name = String::from("optimistic_assert");
    let primop = *isla_primops.binary.get(&primop_name).unwrap();

    let primop_variations = primops
        .binary
        .entry(primop)
        .or_insert_with(|| HashMap::new());
    primop_variations.insert(
        absurd_hash_of_function_params(&param_pointee_tys),
        (primop_name, sail_assert, param_pointee_tys),
    );

    // zsail_assume
    // This doesn't appear anywhere in the Armv9.4a spec
}

#[derive(Debug)]
struct J2LDeclaredTypes<'a> {
    /// Jib makes heavy use of dynamically sized bitvecs. It's the only primitive that is represented as a named LLVM struct, and its representation for
    /// the llvm::Context of 'a is stored here. The Unit type in Jib is treated in LLVM as an empty-length bitvector
    bitvec: llvm::StructType<'a>,
    /// Jib Names to the corresponding LLVM type for any struct, struct field, enum, union, union ctor,
    mappings: HashMap<isla::Name, J2LType<'a>>,
    /// Jib enum member Names to the corresponding LLVM Global constant value representing the member
    enum_members: HashMap<isla::Name, llvm::GlobalValue<'a>>,
    /// Jib struct String name to an ordered dictionary mapping field Names to their offset in the corresponding LLVM struct type
    struct_fields: HashMap<String, BTreeMap<isla::Name, u32>>,
    /// Jib union ctor Name to the Jib union type Name, the LLVM constant Global IntVal representing its tag, and the LLVM type that tag corresponds to
    union_ctors: HashMap<isla::Name, (isla::Name, llvm::GlobalValue<'a>, llvm::BasicTypeEnum<'a>)>,
}

impl<'a> J2LDeclaredTypes<'a> {
    fn new(
        jib_spec: &isla::SharedState<isla::B64>,
        module: &llvm::Module<'a>,
        llvm_ctx: &'a llvm::Context,
    ) -> Self {
        let bitvec_ty = llvm_ctx.opaque_struct_type("bv");
        bitvec_ty.set_body(
            &[
                llvm_ctx.i128_type().as_basic_type_enum(),
                llvm_ctx.i128_type().as_basic_type_enum(),
            ],
            true,
        );

        let mut types = J2LDeclaredTypes {
            bitvec: bitvec_ty,
            mappings: HashMap::new(),
            enum_members: HashMap::new(),
            struct_fields: HashMap::new(),
            union_ctors: HashMap::new(),
        };

        // Enums can be done in a single pass, because they don't contain any other types
        for (jib_id, jib_enum_members) in jib_spec.type_info.enums.iter() {
            let bits_needed = jib_enum_members.len().ilog2() + 1;
            let llvm_ty = llvm_ctx.custom_width_int_type(bits_needed);

            for (i, enum_member) in jib_enum_members.iter().enumerate() {
                let member_val = llvm_ty.const_int(i as u64, false);
                let member = module.add_global(
                    member_val.get_type(),
                    Some(llvm::AddressSpace::default()),
                    &isla::zencode::decode(jib_spec.symtab.to_str_demangled(enum_member.clone())),
                );
                member.set_constant(true);
                member.set_initializer(&member_val);

                let prev = types.enum_members.insert(enum_member.clone(), member);
                assert_eq!(prev, None);
            }

            let prev = types.mappings.insert(
                jib_id.clone(),
                J2LType::Enum(llvm_ty, jib_enum_members.len() as u32),
            );
            assert_eq!(prev, None);
        }

        // Structs and unions need indefinite possible passes, because I don't care to figure out recursion for this
        loop {
            for (jib_id, jib_fields) in jib_spec.type_info.structs.iter() {
                let already_contructed = types.mappings.contains_key(jib_id);
                let can_construct = jib_fields.iter().all(|(_, jib_ty)| {
                    use isla::Ty::*;
                    match jib_ty {
                        Enum(id) => types.mappings.contains_key(id),
                        Struct(id) => types.mappings.contains_key(id),
                        Union(id) => types.mappings.contains_key(id),
                        _ => true,
                    }
                });
                if !already_contructed && can_construct {
                    let name = isla::zencode::decode(jib_spec.symtab.to_str(jib_id.clone()));
                    let llvm_ty = llvm_ctx.opaque_struct_type(&name);
                    let llvm_fields = jib_fields
                        .iter()
                        .map(|(_, field_ty)| j2l_ty(field_ty.clone(), &types, &llvm_ctx))
                        .collect::<Vec<llvm::BasicTypeEnum>>();
                    llvm_ty.set_body(&llvm_fields, true);

                    let prev = types
                        .mappings
                        .insert(jib_id.clone(), J2LType::Struct(llvm_ty));
                    assert_eq!(prev, None);
                    let mut field_mappings = BTreeMap::new();
                    for (i, (jib_id, _)) in jib_fields.iter().enumerate() {
                        let prev = field_mappings.insert(jib_id.clone(), i as u32);
                        assert_eq!(prev, None);
                    }
                    let prev = types.struct_fields.insert(name.to_string(), field_mappings);
                    assert_eq!(prev, None);
                }
            }

            for (jib_id, jib_ctors) in jib_spec.type_info.unions.iter() {
                let already_contructed = types.mappings.contains_key(jib_id);
                let can_construct = jib_ctors.iter().all(|(_, ctor_ty)| {
                    use isla::Ty::*;
                    match ctor_ty {
                        Enum(id) => types.mappings.contains_key(id),
                        Struct(id) => types.mappings.contains_key(id),
                        Union(id) => types.mappings.contains_key(id),
                        _ => true,
                    }
                });
                if !already_contructed && can_construct {
                    let tag_bits_needed = jib_ctors.len().ilog2() + 1;
                    let llvm_tag_ty = llvm_ctx
                        .custom_width_int_type(tag_bits_needed)
                        .as_basic_type_enum();
                    let ctor_bits_needed = jib_ctors
                        .iter()
                        .map(|(_, ctor_ty)| {
                            let llvm_ty = j2l_ty(ctor_ty.clone(), &types, llvm_ctx);
                            llvm_ty_size(llvm_ty)
                        })
                        .max()
                        .unwrap()
                        .ilog2()
                        + 1;
                    for (i, (ctor_id, ctor_ty)) in jib_ctors.iter().enumerate() {
                        let tag_val = llvm_tag_ty.into_int_type().const_int(i as u64, false);
                        let corresponding_ty = j2l_ty(ctor_ty.clone(), &types, llvm_ctx);
                        let ctor = module.add_global(
                            llvm_tag_ty,
                            Some(llvm::AddressSpace::default()),
                            &isla::zencode::decode(
                                jib_spec.symtab.to_str_demangled(ctor_id.clone()),
                            ),
                        );
                        ctor.set_constant(true);
                        ctor.set_initializer(&tag_val);

                        let prev = types
                            .union_ctors
                            .insert(ctor_id.clone(), (jib_id.clone(), ctor, corresponding_ty));
                        assert_eq!(prev, None);
                    }

                    let llvm_ctor_ty = llvm_ctx
                        .custom_width_int_type(ctor_bits_needed)
                        .as_basic_type_enum();

                    let llvm_ty = llvm_ctx.struct_type(&[llvm_tag_ty, llvm_ctor_ty], true);
                    let prev = types
                        .mappings
                        .insert(jib_id.clone(), J2LType::Union(llvm_ty));
                    assert_eq!(prev, None);
                }
            }

            let mappings_created = types.mappings.len();
            let mappings_needed = jib_spec.type_info.enums.len()
                + jib_spec.type_info.structs.len()
                + jib_spec.type_info.unions.len();

            if mappings_created == mappings_needed {
                break;
            }
        }

        types
    }
}

#[derive(Debug, PartialEq, Eq)]
enum J2LType<'a> {
    Enum(llvm::IntType<'a>, u32),
    Struct(llvm::StructType<'a>),
    Union(llvm::StructType<'a>),
}

fn absurd_hash_of_function_params<'a>(params: &[llvm::BasicMetadataTypeEnum<'a>]) -> String {
    params
        .iter()
        .map(|param| format!("{param}"))
        .collect::<Vec<String>>()
        .join("--")
}

type J2LPrimopMap<'a, K> = HashMap<
    K,
    HashMap<
        String,
        (
            String,
            llvm::FunctionValue<'a>,
            Vec<llvm::BasicMetadataTypeEnum<'a>>,
        ),
    >,
>;

#[derive(Debug)]
struct J2LPrimops<'a> {
    unary: J2LPrimopMap<'a, isla::Unary<isla::B64>>,
    binary: J2LPrimopMap<'a, isla::Binary<isla::B64>>,
    variadic: J2LPrimopMap<'a, isla::Variadic<isla::B64>>,
}

impl<'a> J2LPrimops<'a> {
    fn new() -> Self {
        Self {
            unary: HashMap::new(),
            binary: HashMap::new(),
            variadic: HashMap::new(),
        }
    }
}

#[derive(Debug)]
struct J2LLocalContext<'a> {
    /// Args are read-only pointers to the callers stack variables with the exception of the last arguemnt: a pointer
    /// to the return value. This pointer is write-only. Accesses to all args are type-checked at translation-time.
    args: BTreeMap<isla::Name, (llvm::BasicTypeEnum<'a>, llvm::PointerValue<'a>)>,
    /// These are declared locally to the function stack by ::Decl and ::Init instrs. Access are type-checed at translation
    locals: HashMap<isla::Name, (llvm::BasicTypeEnum<'a>, llvm::PointerValue<'a>)>,
}

impl<'a> J2LLocalContext<'a> {
    fn new_val(
        &mut self,
        id: isla::Name,
        ty: llvm::BasicTypeEnum<'a>,
        ptr: llvm::PointerValue<'a>,
    ) {
        let prev = self.locals.insert(id, (ty, ptr));
        assert_eq!(prev, None);
    }
}

fn unssa_ty(ty: &isla::Ty<isla::SSAName>) -> isla::Ty<isla::Name> {
    use isla::Ty::*;
    match ty {
        I64 => I64,
        I128 => I128,
        AnyBits => AnyBits,
        Bits(n) => Bits(*n),
        Unit => Unit,
        Bool => Bool,
        Bit => Bit,
        String => String,
        Real => Real,
        Enum(id) => {
            assert!(id.clone().ssa_number() < 0);
            Enum(id.base_name())
        }
        Struct(id) => {
            assert!(id.clone().ssa_number() < 0);
            Struct(id.base_name())
        }
        Union(id) => {
            assert!(id.clone().ssa_number() < 0);
            Union(id.base_name())
        }
        Vector(ty) => Vector(Box::new(unssa_ty(ty))),
        FixedVector(n, ty) => FixedVector(*n, Box::new(unssa_ty(ty))),
        List(ty) => List(Box::new(unssa_ty(ty))),
        Ref(ty) => Ref(Box::new(unssa_ty(ty))),
        Float(fpty) => Float(*fpty),
        RoundingMode => RoundingMode,
    }
}

fn llvm_ty_size(ty: llvm::BasicTypeEnum) -> u32 {
    use llvm::BasicTypeEnum::*;
    match ty {
        IntType(ty) => ty.get_bit_width(),
        FloatType(ty) => {
            // let sz = ty.size_of().get_zero_extended_constant().unwrap() as u32;
            // assert!(sz == 16 || sz == 32 || sz == 64);
            // sz
            128
        }
        ArrayType(ty) => ty.len() * llvm_ty_size(ty.get_element_type().as_basic_type_enum()),
        StructType(ty) => {
            let mut sum = 0;
            for field in ty.get_field_types_iter() {
                sum += llvm_ty_size(field.as_basic_type_enum());
            }
            sum
        }
        VectorType(ty) => ty.get_size(),
        PointerType(ty) => 64, // TODO: should be real
    }
}

fn llvm_deep_eq<'a>(
    lhs: llvm::BasicValueEnum<'a>,
    rhs: llvm::BasicValueEnum<'a>,
    builder: &llvm::Builder<'a>,
    llvm_ctx: &'a llvm::Context,
) -> llvm::IntValue<'a> {
    use llvm::BasicTypeEnum::*;

    assert_eq!(lhs.get_type(), rhs.get_type());
    let result = match lhs.get_type() {
        IntType(_) => builder
            .build_int_compare(
                llvm::IntPredicate::EQ,
                lhs.into_int_value(),
                rhs.into_int_value(),
                "int_eq",
            )
            .unwrap(),
        PointerType(_) => builder
            .build_int_compare(
                llvm::IntPredicate::EQ,
                lhs.into_pointer_value(),
                rhs.into_pointer_value(),
                "ptr_eq",
            )
            .unwrap(),
            FloatType(_) => builder.build_float_compare(llvm::FloatPredicate::OEQ, lhs.into_float_value(), rhs.into_float_value(), "float_eq").unwrap(),
            StructType(_) => {
            let lhs = lhs.into_struct_value();
            let rhs = rhs.into_struct_value();
            let mut all_prev_are_equal = llvm_ctx.bool_type().const_all_ones();
            for (l_field, r_field) in lhs.get_fields().zip(rhs.get_fields()) {
                let fields_are_eq = llvm_deep_eq(l_field, r_field, builder, llvm_ctx);
                all_prev_are_equal = builder.build_and(all_prev_are_equal, fields_are_eq, "field_eq").unwrap();
            }
            all_prev_are_equal
        },
        ty => unreachable!("All Jib types map to one of these LLVM types (except array, is that what is used here? {ty}).")
    };
    result
}

enum Assign {
    Nop,
    Pointer,
    Pointee,
}

fn llvm_assign_chk<'a>(dst: llvm::BasicTypeEnum<'a>, src: llvm::BasicTypeEnum<'a>) -> Assign {
    if let llvm::BasicTypeEnum::ArrayType(dst) = dst {
        if let llvm::BasicTypeEnum::ArrayType(src) = src {
            assert_eq!(dst.get_element_type(), src.get_element_type());

            // Assign global array to local zero-length array
            if src.len() > dst.len() {
                return Assign::Pointer;
            } else {
                // Assign local zero-length array that Jib thinks was modified by a vector primop
                // Really, these modify the global arrays in-place so nothing should happen at these
                // assignments.
                return Assign::Nop;
            }
        }
    }
    // if let llvm::BasicTypeEnum::PointerType(dst) = dst {
    //     if let llvm::BasicTypeEnum::ArrayType(src) = src {
    //         // Assign global array to local zero-length array
    //         return Assign::Pointer;
    //     }
    // }

    // if let llvm::BasicTypeEnum::ArrayType(dst) = dst {
    //     if let llvm::BasicTypeEnum::PointerType(src) = src {
    //         // Assign local zero-length array that Jib thinks was modified by a vector primop
    //         // Really, these modify the global arrays in-place so nothing should happen at these
    //         // assignments.
    //         return Assign::Nop;
    //     }
    // }

    assert_eq!(dst, src);
    return Assign::Pointee;
}

fn llvm_call_arg_chk<'a>(param: llvm::BasicTypeEnum<'a>, arg: llvm::BasicTypeEnum<'a>) {
    if let llvm::BasicTypeEnum::ArrayType(arg) = arg {
        if let llvm::BasicTypeEnum::ArrayType(param) = param {
            assert_eq!(arg.get_element_type(), param.get_element_type());
        }
    }
    assert_eq!(param, arg);
}

fn llvm_set_fn_attrs<'a>(
    f: &llvm::FunctionValue<'a>,
    param_pointee_tys: &[llvm::BasicMetadataTypeEnum],
    llvm_ctx: &'a llvm::Context,
) {
    for (param_idx, param_pointee_ty) in param_pointee_tys.iter().enumerate() {
        let param_pointee_ty = match param_pointee_ty {
            llvm::BasicMetadataTypeEnum::ArrayType(ty) => llvm::AnyTypeEnum::from(*ty),
            llvm::BasicMetadataTypeEnum::FloatType(ty) => llvm::AnyTypeEnum::from(*ty),
            llvm::BasicMetadataTypeEnum::IntType(ty) => llvm::AnyTypeEnum::from(*ty),
            llvm::BasicMetadataTypeEnum::StructType(ty) => llvm::AnyTypeEnum::from(*ty),
            llvm::BasicMetadataTypeEnum::PointerType(ty) => llvm::AnyTypeEnum::from(*ty),
            llvm::BasicMetadataTypeEnum::VectorType(ty) => llvm::AnyTypeEnum::from(*ty),
            llvm::BasicMetadataTypeEnum::MetadataType(ty) => unreachable!(),
        };

        let attr;
        if param_idx == 0 {
            let sret_attr = llvm::Attribute::get_named_enum_kind_id("sret");
            attr = llvm_ctx.create_type_attribute(sret_attr, param_pointee_ty);
        } else {
            let byref_attr = llvm::Attribute::get_named_enum_kind_id("byref");
            attr = llvm_ctx.create_type_attribute(byref_attr, param_pointee_ty);
        }
        f.add_attribute(llvm::AttributeLoc::Param(param_idx as u32), attr);
    }
}

fn llvm_bv_bits<'a>(
    ty: llvm::StructType<'a>,
    ptr: llvm::PointerValue<'a>,
    builder: &llvm::Builder<'a>,
    llvm_ctx: &'a llvm::Context,
) -> llvm::IntValue<'a> {
    let len_ptr = builder.build_struct_gep(ty, ptr, 0, "len_ptr").unwrap();
    let len = builder
        .build_load(llvm_ctx.i128_type(), len_ptr, "len")
        .unwrap()
        .into_int_value()
        .get_zero_extended_constant()
        .expect("j2l_exp can only produce const bitvecs");
    let bits_ptr = builder.build_struct_gep(ty, ptr, 1, "bits_ptr").unwrap();
    let bits = builder
        .build_load(llvm_ctx.i128_type(), bits_ptr, "bits")
        .unwrap()
        .into_int_value();
    let bits_sized = builder
        .build_int_truncate(
            bits,
            llvm_ctx.custom_width_int_type(len as u32),
            "bits_sized",
        )
        .unwrap();
    bits_sized
}

/// Define a new LLVM equivalent to a Jib register (mutable global variable)
fn j2l_register_decl<'a>(
    name: &str,
    ty: isla::Ty<isla::Name>,
    types: &J2LDeclaredTypes<'a>,
    module: &llvm::Module<'a>,
    llvm_ctx: &'a llvm::Context,
) -> llvm::GlobalValue<'a> {
    let ty = if let isla::Ty::Vector(elem_ty) = ty {
        panic!("Tried to declare an arbitrary length global vector: {name} %vec({elem_ty:?})");
    } else {
        j2l_ty(ty, types, llvm_ctx)
    };
    let ty = llvm::BasicTypeEnum::try_from(ty).unwrap();
    let glob = module.add_global(ty, Some(llvm::AddressSpace::default()), &name);
    glob
}

/// Convert a Jib type to an LLVM equivalent
fn j2l_ty<'a>(
    ty: isla::Ty<isla::Name>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) -> llvm::BasicTypeEnum<'a> {
    use isla::Ty::*;
    match ty {
        Bool => llvm_ctx.bool_type().as_basic_type_enum(),
        I64 => llvm_ctx.i64_type().as_basic_type_enum(),
        I128 => llvm_ctx.i128_type().as_basic_type_enum(),
        // Jib bitvectors are a struct of length and value.
        // Unit types can be declared and assigned in Jib, so we can't treat them like LLVM void types which forbid those behaviors.
        // Instead, we will treat them as zero-length bitvectors.
        AnyBits | Bits(_) | Bit | Unit => types.bitvec.as_basic_type_enum(),
        // All floats and real numbers are treated as 128-bit for now, even though Jib only supports standard IEEE floats.
        // This is because Inkwell doesn't give a method for determining the bit-widtth of a float (seems odd).
        Float(_) | Real => {
            llvm_ctx.f128_type().as_basic_type_enum()
            //     match (ty.exponent_width(), ty.significand_width()) {
            //     (5, 11) => llvm_ctx.f16_type().as_basic_type_enum(),
            //     (8, 24) => llvm_ctx.f32_type().as_basic_type_enum(),
            //     (11, 53) => llvm_ctx.f64_type().as_basic_type_enum(),
            //     (15, 113) => llvm_ctx.f128_type().as_basic_type_enum(),
            //     (e, s) => panic!("Unsupported FPTy: {0} bits", e + s),
            // }
        }
        // Named types (enums, structs, and unions) are already contained in the LLVMTypeInfo struct.
        Enum(id) => {
            if let J2LType::Enum(llvm_ty, _) = types.mappings.get(&id).unwrap() {
                llvm_ty.clone().as_basic_type_enum()
            } else {
                unreachable!()
            }
        }
        Struct(id) => {
            if let J2LType::Struct(llvm_ty) = types.mappings.get(&id).unwrap() {
                llvm_ty.clone().as_basic_type_enum()
            } else {
                unreachable!()
            }
        }
        Union(id) => {
            if let J2LType::Union(llvm_ty) = types.mappings.get(&id).unwrap() {
                llvm_ty.clone().as_basic_type_enum()
            } else {
                unreachable!()
            }
        }
        Vector(elem_ty) => match j2l_ty(*elem_ty, types, llvm_ctx) {
            llvm::BasicTypeEnum::IntType(ty) => ty.array_type(0).as_basic_type_enum(),
            llvm::BasicTypeEnum::FloatType(ty) => ty.array_type(0).as_basic_type_enum(),
            llvm::BasicTypeEnum::PointerType(ty) => ty.array_type(0).as_basic_type_enum(),
            llvm::BasicTypeEnum::StructType(ty) => ty.array_type(0).as_basic_type_enum(),
            other => panic!("Unsupported type {other}"),
        },
        FixedVector(len, elem_ty) => match j2l_ty(*elem_ty, types, llvm_ctx) {
            llvm::BasicTypeEnum::IntType(ty) => ty.array_type(len).as_basic_type_enum(),
            llvm::BasicTypeEnum::FloatType(ty) => ty.array_type(len).as_basic_type_enum(),
            llvm::BasicTypeEnum::PointerType(ty) => ty.array_type(len).as_basic_type_enum(),
            llvm::BasicTypeEnum::StructType(ty) => ty.array_type(len).as_basic_type_enum(),
            other => panic!("Unsupported type {other}"),
        },
        // Strings are byte-arrays
        String => llvm_ctx.i8_type().array_type(0).as_basic_type_enum(),
        List(ty) => {
            // Only known usage of List type in the Armv9.4a spec is as an argument to "internal_pick" functions
            // which in turn are only used in selecting a variant for unintialized enums in "undefined_x" functions.
            // Therefore, a list of enums will just be a single enum value and internal_pick will just return
            // that
            if let Enum(elem_id) = *ty {
                if let J2LType::Enum(llvm_ty, _num_variants) = types.mappings.get(&elem_id).unwrap()
                {
                    // return llvm_ty.array_type(*num_variants).as_basic_type_enum();
                    return llvm_ty.as_basic_type_enum();
                }
            }
            todo!("List of {ty:?}")
        }
        Ref(_ty) => llvm_ctx
            .ptr_type(llvm::AddressSpace::default())
            .as_basic_type_enum(),
        RoundingMode => todo!("RoundingMode"),
    }
}

fn j2l_id<'a>(
    id: &isla::SSAName,
    locals: &J2LLocalContext<'a>,
    statics: &J2LStaticContext<'a>,
) -> (llvm::BasicTypeEnum<'a>, llvm::PointerValue<'a>) {
    let base_name = id.clone().base_name();
    if let Some(reg) = statics.registers.get(&base_name) {
        let pointee_ty = llvm::BasicTypeEnum::try_from(reg.get_value_type()).unwrap();
        (pointee_ty, reg.as_pointer_value())
    } else if let Some(let_binding) = statics.lets.get(&base_name) {
        let pointee_ty = llvm::BasicTypeEnum::try_from(let_binding.get_value_type()).unwrap();
        (pointee_ty, let_binding.as_pointer_value())
    } else if let Some(member) = statics.types.enum_members.get(&base_name) {
        let pointee_ty = llvm::BasicTypeEnum::try_from(member.get_value_type()).unwrap();
        (pointee_ty, member.as_pointer_value())
    } else if let Some((_, tag, _)) = statics.types.union_ctors.get(&base_name) {
        let pointee_ty = llvm::BasicTypeEnum::try_from(tag.get_value_type()).unwrap();
        (pointee_ty, tag.as_pointer_value())
    } else {
        locals
            .args
            .get(&base_name)
            .or_else(|| locals.locals.get(&base_name))
            .unwrap_or_else(|| panic!("Couldn't find val {}", statics.ssa_name(id)))
            .clone()
    }
}

fn j2l_static_val<'a>(
    val: &isla::Val<isla::B64>,
    types: &J2LDeclaredTypes<'a>,
    registers: &HashMap<isla::Name, llvm::GlobalValue<'a>>,
    llvm_ctx: &'a llvm::Context,
) -> llvm::BasicValueEnum<'a> {
    use isla::Val::*;
    match val {
        Symbolic(_) | MixedBits(..) | SymbolicCtor(..) | Poison => {
            llvm_ctx.bool_type().get_poison().as_basic_value_enum()
        }
        I64(val) => {
            let ty = llvm_ctx.i64_type();
            let init = ty.const_int(u64::from_le_bytes(val.to_le_bytes()), true);
            init.as_basic_value_enum()
        }
        I128(val) => {
            let ty = llvm_ctx.i128_type();

            // Get val as two 64-bit words for the LLVM API
            let bytes = val.to_le_bytes();
            let mut lo64 = [0; 8];
            lo64.copy_from_slice(&bytes[0..8]);
            let lo64 = u64::from_le_bytes(lo64);
            let mut hi64 = [0; 8];
            hi64.copy_from_slice(&bytes[8..16]);
            let hi64 = u64::from_le_bytes(hi64);
            let val = ty.const_int_arbitrary_precision(&[lo64, hi64]);
            val.as_basic_value_enum()
        }
        Bool(b) => {
            let ty = llvm_ctx.bool_type();
            let init = ty.const_int(if *b { 1 } else { 0 }, false);
            init.as_basic_value_enum()
        }
        Bits(val) => {
            let len = llvm_ctx.i128_type().const_int(val.len() as u64, false);
            let bits = llvm_ctx.i128_type().const_int(val.lower_u64(), false);
            types
                .bitvec
                .const_named_struct(&[len.as_basic_value_enum(), bits.as_basic_value_enum()])
                .as_basic_value_enum()
        }
        String(s) => unimplemented!("String"),
        Unit => {
            let len = llvm_ctx.i128_type().const_int(0, false);
            let bits = llvm_ctx.i128_type().const_int(0, false);
            types
                .bitvec
                .const_named_struct(&[len.as_basic_value_enum(), bits.as_basic_value_enum()])
                .as_basic_value_enum()
        }
        Ref(reg) => registers
            .get(reg)
            .unwrap()
            .as_pointer_value()
            .as_basic_value_enum(),
        Vector(v) => {
            let val;
            if let Ref(_) = &v[0] {
                let elem_ty = llvm_ctx.ptr_type(llvm::AddressSpace::default());

                let values = v
                    .iter()
                    .map(|e| match e {
                        Ref(e) => registers.get(e).unwrap().as_pointer_value(),
                        other => unreachable!("{other:?}"),
                    })
                    .collect::<Vec<llvm::PointerValue>>();
                val = elem_ty.const_array(&values);
            } else if let Bits(_) = &v[0] {
                let elem_ty = types.bitvec.clone();

                let values = v
                    .iter()
                    .map(|e| match e {
                        Bits(val) => {
                            let len = llvm_ctx.i128_type().const_int(val.len() as u64, false);
                            let bits = llvm_ctx.i128_type().const_int(val.lower_u64(), false);
                            types.bitvec.const_named_struct(&[
                                len.as_basic_value_enum(),
                                bits.as_basic_value_enum(),
                            ])
                        }
                        other => unreachable!("{other:?}"),
                    })
                    .collect::<Vec<llvm::StructValue>>();
                val = elem_ty.const_array(&values);
            } else {
                let elem_ty = match &v[0] {
                    I64(_) => llvm_ctx.i64_type(),
                    I128(_) => llvm_ctx.i128_type(),
                    Bool(_) => llvm_ctx.bool_type(),
                    other => unimplemented!("{other:?}"),
                };

                let values = v
                    .iter()
                    .map(|e| match e {
                        I64(val) => elem_ty.const_int(u64::from_le_bytes(val.to_le_bytes()), true),
                        I128(val) => {
                            // Get val as two 64-bit words for the LLVM API
                            let bytes = val.to_le_bytes();
                            let mut lo64 = [0; 8];
                            lo64.copy_from_slice(&bytes[0..8]);
                            let lo64 = u64::from_le_bytes(lo64);
                            let mut hi64 = [0; 8];
                            hi64.copy_from_slice(&bytes[8..16]);
                            let hi64 = u64::from_le_bytes(hi64);
                            elem_ty.const_int_arbitrary_precision(&[lo64, hi64])
                        }
                        Bool(val) => elem_ty.const_int(if *val { 1 } else { 0 }, false),
                        other => unimplemented!("{other:?}"),
                    })
                    .collect::<Vec<llvm::IntValue>>();
                val = elem_ty.const_array(&values);
            }
            val.as_basic_value_enum()
        }
        List(l) => unimplemented!("List"),
        Enum(member) => unimplemented!("Enum"),
        Struct(fields) => unimplemented!("Struct"),
        Ctor(tag, val) => {
            let (parent, tag, ty) = types.union_ctors.get(tag).unwrap();
            let parent_ty = if let J2LType::Union(ty) = types.mappings.get(parent).unwrap() {
                ty.clone()
            } else {
                panic!("Unknown union type {parent:?}");
            };
            let tag = tag.get_initializer().unwrap().into_int_value();
            let val = j2l_static_val(val, types, registers, llvm_ctx);
            parent_ty
                .const_named_struct(&[tag.as_basic_value_enum(), val])
                .as_basic_value_enum()
        }
    }
}

/// Evaluate a Jib expression to an LLVM value, creating a new value if needed, and return a pointer to the expression result
fn j2l_exp<'a>(
    exp: &isla::Exp<isla::SSAName>,
    locals: &J2LLocalContext<'a>,
    statics: &J2LStaticContext<'a>,
    strings: &mut HashMap<String, llvm::GlobalValue<'a>>,
    builder: &llvm::Builder<'a>,
    module: &llvm::Module<'a>,
    llvm_ctx: &'a llvm::Context,
) -> (llvm::BasicTypeEnum<'a>, llvm::PointerValue<'a>) {
    use isla::Exp::*;
    match exp {
        Id(id) => j2l_id(id, locals, statics),
        // This could just be returning the pointer same as j2l_id. Where things could get tricky is if
        // this reference is meant to be mutable. All other pointers are assumed to be immutable.
        // The only known instance of Ref expressions is to pass the function pointer of an abstract function as an argument
        // to the "call abstract function" handling function that is built-in to isla. In initialize_architecture, isla replaces
        // all calls to abstract functions with a call to a handler identified by ABSTRACT_PRIMOP and adds the pointer to the
        // target abstract function as an argument to the variadic args of ABSTRACT_PRIMOP. Since this is the only case we know
        // about for now, a Ref will be a pointer to a function.
        Ref(id) => {
            let (abstract_fn, _param_pointee_tys) = statics.get_fn(&id.base_name()).unwrap();
            let ty = llvm_ctx
                .ptr_type(llvm::AddressSpace::default())
                .as_basic_type_enum();
            let val = abstract_fn.as_global_value().as_pointer_value();
            (ty, val)
        }
        I64(val) => {
            let ty = llvm_ctx.i64_type();
            let ptr = builder.build_alloca(ty, "const").unwrap();
            let val = ty.const_int(u64::from_le_bytes(val.to_le_bytes()), true);
            builder.build_store(ptr, val).unwrap();
            (ty.as_basic_type_enum(), ptr)
        }
        I128(val) => {
            let ty = llvm_ctx.i128_type();
            let ptr = builder.build_alloca(ty, "const").unwrap();

            // Get val as two 64-bit words for the LLVM API
            let bytes = val.to_le_bytes();
            let mut lo64 = [0; 8];
            lo64.copy_from_slice(&bytes[0..8]);
            let lo64 = u64::from_le_bytes(lo64);
            let mut hi64 = [0; 8];
            hi64.copy_from_slice(&bytes[8..16]);
            let hi64 = u64::from_le_bytes(hi64);
            let val = ty.const_int_arbitrary_precision(&[lo64, hi64]);

            builder.build_store(ptr, val).unwrap();
            (ty.as_basic_type_enum(), ptr)
        }
        Unit => {
            let ty = statics.types.bitvec.clone();
            let ptr = builder.build_alloca(ty, "const").unwrap();

            let len = llvm_ctx.i128_type().const_int(0, false);
            let bits = llvm_ctx.i128_type().const_int(0, false);
            let val =
                ty.const_named_struct(&[len.as_basic_value_enum(), bits.as_basic_value_enum()]);

            builder.build_store(ptr, val).unwrap();
            (ty.as_basic_type_enum(), ptr)
        }
        Bool(val) => {
            let ty = llvm_ctx.bool_type();
            let ptr = builder.build_alloca(ty, "const").unwrap();
            let val = ty.const_int(if *val { 1 } else { 0 }, false);
            builder.build_store(ptr, val).unwrap();
            (ty.as_basic_type_enum(), ptr)
        }
        Bits(val) => {
            let ty = statics.types.bitvec.clone();
            let ptr = builder.build_alloca(ty, "const").unwrap();

            let len = llvm_ctx.i128_type().const_int(val.len() as u64, false);
            let bits = llvm_ctx.i128_type().const_int(val.lower_u64(), false);
            let val =
                ty.const_named_struct(&[len.as_basic_value_enum(), bits.as_basic_value_enum()]);

            builder.build_store(ptr, val).unwrap();
            (ty.as_basic_type_enum(), ptr)
        }
        String(s) => {
            let global = strings.entry(s.clone()).or_insert_with(|| {
                let c_str = CString::new(s.clone()).unwrap();

                let global = module.add_global(
                    llvm_ctx
                        .i8_type()
                        .array_type((c_str.count_bytes() + 1) as u32),
                    Some(llvm::AddressSpace::default()),
                    "static_string",
                );

                let value = llvm_ctx.i8_type().const_array(
                    &c_str
                        .as_bytes_with_nul()
                        .iter()
                        .map(|ascii_char| llvm_ctx.i8_type().const_int(*ascii_char as u64, false))
                        .collect::<Vec<llvm::IntValue>>(),
                );

                global.set_initializer(&value);
                global.set_constant(true);
                global
            });

            // let ty = global.get_value_type();
            // let ty = llvm::BasicTypeEnum::try_from(ty).unwrap();
            let ty = llvm_ctx.i8_type().array_type(0).as_basic_type_enum();
            (ty, global.as_pointer_value())
        }
        Undefined(ty) => (
            j2l_ty(unssa_ty(ty), &statics.types, llvm_ctx),
            llvm_ctx
                .ptr_type(llvm::AddressSpace::default())
                .const_null(),
        ),
        Call(op, args) => {
            let args = args
                .iter()
                .map(|arg| j2l_exp(arg, locals, statics, strings, builder, module, llvm_ctx))
                .collect::<Vec<(llvm::BasicTypeEnum, llvm::PointerValue)>>();

            use isla::Op::*;
            match op {
                Lt | Gt | Lteq | Gteq => {
                    let lhs = builder
                        .build_load(args[0].0.into_int_type(), args[0].1, "cmp_lhs")
                        .unwrap()
                        .into_int_value();
                    let rhs = builder
                        .build_load(args[1].0.into_int_type(), args[1].1, "cmp_rhs")
                        .unwrap()
                        .into_int_value();
                    assert_eq!(lhs.get_type(), rhs.get_type());
                    let cmp_result = match op {
                        Lt => builder
                            .build_int_compare(llvm::IntPredicate::SLT, lhs, rhs, "cmp_res")
                            .unwrap(),
                        Gt => builder
                            .build_int_compare(llvm::IntPredicate::SGT, lhs, rhs, "cmp_res")
                            .unwrap(),
                        Lteq => builder
                            .build_int_compare(llvm::IntPredicate::SLE, lhs, rhs, "cmp_res")
                            .unwrap(),
                        Gteq => builder
                            .build_int_compare(llvm::IntPredicate::SGE, lhs, rhs, "cmp_res")
                            .unwrap(),
                        _ => unreachable!(),
                    };
                    let result_val = builder
                        .build_alloca(llvm_ctx.bool_type(), "cmp_res_val")
                        .unwrap();
                    builder.build_store(result_val, cmp_result).unwrap();
                    (cmp_result.get_type().as_basic_type_enum(), result_val)
                }
                Eq | Neq => {
                    let lhs_ty = llvm::BasicTypeEnum::try_from(args[0].0).unwrap();
                    let rhs_ty = llvm::BasicTypeEnum::try_from(args[1].0).unwrap();
                    let lhs = builder.build_load(lhs_ty, args[0].1, "eq_lhs").unwrap();
                    let rhs = builder.build_load(rhs_ty, args[1].1, "eq_rhs").unwrap();

                    let eq_result = llvm_deep_eq(lhs, rhs, builder, llvm_ctx);
                    let eq_result = if let Neq = op {
                        builder.build_not(eq_result, "not_eq").unwrap()
                    } else {
                        eq_result
                    };
                    let result_val = builder
                        .build_alloca(llvm_ctx.bool_type(), "eq_res_val")
                        .unwrap();

                    builder.build_store(result_val, eq_result).unwrap();
                    (eq_result.get_type().as_basic_type_enum(), result_val)
                }
                Add | Sub | Or | And => {
                    let lhs = builder
                        .build_load(args[0].0.into_int_type(), args[0].1, "lhs")
                        .unwrap()
                        .into_int_value();
                    let rhs = builder
                        .build_load(args[1].0.into_int_type(), args[1].1, "rhs")
                        .unwrap()
                        .into_int_value();
                    assert_eq!(lhs.get_type(), rhs.get_type());
                    let arith_result = match op {
                        Add => builder.build_int_add(lhs, rhs, "res").unwrap(),
                        Sub => builder.build_int_sub(lhs, rhs, "res").unwrap(),
                        Or => builder.build_or(lhs, rhs, "res").unwrap(),
                        And => builder.build_and(lhs, rhs, "res").unwrap(),
                        _ => unreachable!(),
                    };
                    let result_val = builder.build_alloca(lhs.get_type(), "res_val").unwrap();
                    builder.build_store(result_val, arith_result).unwrap();
                    (arith_result.get_type().as_basic_type_enum(), result_val)
                }
                Bvadd | Bvsub | Bvor | Bvxor | Bvand => {
                    let lhs =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, llvm_ctx);
                    let rhs =
                        llvm_bv_bits(args[1].0.into_struct_type(), args[1].1, builder, llvm_ctx);
                    assert_eq!(lhs.get_type(), rhs.get_type());

                    let arith_result = match op {
                        Bvadd => builder.build_int_add(lhs, rhs, "res").unwrap(),
                        Bvsub => builder.build_int_sub(lhs, rhs, "res").unwrap(),
                        Bvor => builder.build_or(lhs, rhs, "res").unwrap(),
                        Bvxor => builder.build_xor(lhs, rhs, "res").unwrap(),
                        Bvand => builder.build_and(lhs, rhs, "res").unwrap(),
                        _ => unreachable!(),
                    };
                    let result_val = builder.build_alloca(lhs.get_type(), "res_val").unwrap();
                    builder.build_store(result_val, arith_result).unwrap();
                    (arith_result.get_type().as_basic_type_enum(), result_val)
                }
                Not => {
                    let val = builder
                        .build_load(args[0].0.into_int_type(), args[0].1, "not_arg")
                        .unwrap()
                        .into_int_value();
                    let result = builder.build_not(val, "not_res").unwrap();
                    let result_val = builder.build_alloca(val.get_type(), "not_res_val").unwrap();
                    builder.build_store(result_val, result).unwrap();
                    (val.get_type().as_basic_type_enum(), result_val)
                }
                Bvnot => {
                    let val =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, llvm_ctx);
                    let result = builder.build_not(val, "not_res").unwrap();
                    let result_val = builder.build_alloca(val.get_type(), "not_res_val").unwrap();
                    builder.build_store(result_val, result).unwrap();
                    (val.get_type().as_basic_type_enum(), result_val)
                }
                Bvaccess => {
                    let val =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, &llvm_ctx);
                    // In Jib, this always comes out to 128-bits
                    let n = builder
                        .build_load(args[1].0.into_int_type(), args[1].1, "n")
                        .unwrap()
                        .into_int_value();
                    // No type-check
                    let result_bit = builder
                        .build_right_shift(val, n, false, "res_shifted")
                        .unwrap();
                    let one = result_bit.get_type().const_int(1, false);
                    let result_bit = builder.build_and(result_bit, one, "res_final").unwrap();

                    let bv_ty = statics.types.bitvec.clone();
                    let result_bv_len = llvm_ctx.i128_type().const_int(1, false);
                    let result_bv_bits = builder
                        .build_int_z_extend(result_bit, llvm_ctx.i128_type(), "res_bv_bits")
                        .unwrap();
                    let result_bv = bv_ty.const_named_struct(&[
                        result_bv_len.as_basic_value_enum(),
                        result_bv_bits.as_basic_value_enum(),
                    ]);

                    let result_val = builder.build_alloca(bv_ty, "res_val").unwrap();
                    builder.build_store(result_val, result_bv).unwrap();
                    (result_bv.get_type().as_basic_type_enum(), result_val)
                }
                Concat => {
                    let lower =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, &llvm_ctx);
                    let upper =
                        llvm_bv_bits(args[1].0.into_struct_type(), args[1].1, builder, &llvm_ctx);
                    let result_sz =
                        lower.get_type().get_bit_width() + upper.get_type().get_bit_width();
                    let lower_ext = builder
                        .build_int_z_extend(
                            lower,
                            llvm_ctx.custom_width_int_type(result_sz),
                            "lower_ext",
                        )
                        .unwrap();
                    let upper_ext = builder
                        .build_int_z_extend(
                            upper,
                            llvm_ctx.custom_width_int_type(result_sz),
                            "upper_ext",
                        )
                        .unwrap();
                    let upper_sh = builder
                        .build_left_shift(
                            upper_ext,
                            upper_ext
                                .get_type()
                                .const_int(lower.get_type().get_bit_width() as u64, false),
                            "upper_sh",
                        )
                        .unwrap();
                    let result_bits = builder.build_or(lower_ext, upper_sh, "res").unwrap();

                    let bv_ty = statics.types.bitvec.clone();
                    let result_bv_len = llvm_ctx.i128_type().const_int(result_sz as u64, false);
                    let result_bv_bits = builder
                        .build_int_z_extend(result_bits, llvm_ctx.i128_type(), "res_bv_bits")
                        .unwrap();
                    let result_bv = bv_ty.const_named_struct(&[
                        result_bv_len.as_basic_value_enum(),
                        result_bv_bits.as_basic_value_enum(),
                    ]);

                    let result_val = builder.build_alloca(bv_ty, "res_val").unwrap();
                    builder.build_store(result_val, result_bv).unwrap();
                    (bv_ty.as_basic_type_enum(), result_val)
                }
                Slice(len) => {
                    let bits =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, &llvm_ctx);
                    let from =
                        llvm_bv_bits(args[1].0.into_struct_type(), args[1].1, builder, &llvm_ctx);

                    let mask = (1 << *len) - 1;
                    let to_mask = builder
                        .build_right_shift(bits, from, false, "to_mask")
                        .unwrap();
                    let result_bits = builder
                        .build_and(to_mask, to_mask.get_type().const_int(mask, false), "res")
                        .unwrap();

                    let bv_ty = statics.types.bitvec.clone();
                    let result_bv_len = llvm_ctx.i128_type().const_int(1, false);
                    let result_bv_bits = builder
                        .build_int_z_extend(result_bits, llvm_ctx.i128_type(), "res_bv_bits")
                        .unwrap();
                    let result_bv = bv_ty.const_named_struct(&[
                        result_bv_len.as_basic_value_enum(),
                        result_bv_bits.as_basic_value_enum(),
                    ]);

                    let result_val = builder.build_alloca(bv_ty, "res_val").unwrap();
                    builder.build_store(result_val, result_bv).unwrap();
                    (bv_ty.as_basic_type_enum(), result_val)
                }
                SetSlice => {
                    let bits =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, &llvm_ctx);
                    let n = builder
                        .build_load(args[1].0.into_int_type(), args[1].1, "n")
                        .unwrap()
                        .into_int_value();
                    let update =
                        llvm_bv_bits(args[2].0.into_struct_type(), args[2].1, builder, &llvm_ctx);

                    /*
                     * From "Bit Twiddling Hacks" by Sean Eron Anderson (https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge)
                     *
                     * unsigned int a;    // value to merge in non-masked bits
                     * unsigned int b;    // value to merge in masked bits
                     * unsigned int mask; // 1 where bits from b should be selected; 0 where from a.
                     * unsigned int r;    // result of (a & ~mask) | (b & mask) goes here
                     *
                     * r = a ^ ((a ^ b) & mask);
                     */
                    // mask = ones(update.len) << n
                    let mask = builder
                        .build_left_shift(update.get_type().const_all_ones(), n, "mask")
                        .unwrap();
                    let b = builder.build_left_shift(update, n, "b").unwrap();
                    let a = bits;
                    let a_x_b = builder.build_xor(a, b, "a^b").unwrap();
                    let a_x_b_masked = builder.build_and(a_x_b, mask, "(a^b) & mask").unwrap();
                    let result_bits = builder.build_xor(a, a_x_b_masked, "res").unwrap();

                    let bv_ty = statics.types.bitvec.clone();
                    let result_bv_len = llvm_ctx.i128_type().const_int(1, false);
                    let result_bv_bits = builder
                        .build_int_z_extend(result_bits, llvm_ctx.i128_type(), "res_bv_bits")
                        .unwrap();
                    let result_bv = bv_ty.const_named_struct(&[
                        result_bv_len.as_basic_value_enum(),
                        result_bv_bits.as_basic_value_enum(),
                    ]);

                    let result_val = builder.build_alloca(bv_ty, "res_val").unwrap();
                    builder.build_store(result_val, result_bv).unwrap();
                    (bv_ty.as_basic_type_enum(), result_val)
                }
                Unsigned(_) => {
                    let bits =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, &llvm_ctx);
                    let result = builder
                        .build_int_z_extend(bits, llvm_ctx.i128_type(), "unsigned_arg")
                        .unwrap();
                    let result_val = builder
                        .build_alloca(llvm_ctx.i128_type(), "unsigned_res_val")
                        .unwrap();
                    builder.build_store(result_val, result).unwrap();
                    (llvm_ctx.i128_type().as_basic_type_enum(), result_val)
                }
                Signed(_) => {
                    let bits =
                        llvm_bv_bits(args[0].0.into_struct_type(), args[0].1, builder, &llvm_ctx);
                    let result = builder
                        .build_int_s_extend(bits, llvm_ctx.i128_type(), "signed_arg")
                        .unwrap();
                    let result_val = builder
                        .build_alloca(llvm_ctx.i128_type(), "signed_res_val")
                        .unwrap();
                    builder.build_store(result_val, result).unwrap();
                    (llvm_ctx.i128_type().as_basic_type_enum(), result_val)
                }
                // The semantics for Head and Tail ops in isla's executor aren't what I would exepect.
                // Both operate on the List type which is represented as a Vec of Jib Val's. op_head pops the last element of that Vec
                // off and returns it. op_tail also pops the last element of the Vec off but then returns the modified list. These may
                // be mistakes in isla that don't really matter, because lists are seldom used. j2l will do what I expect head and list
                // to mean with the head being element 0.
                Head => {
                    let val = builder
                        .build_load(args[0].0.into_array_type(), args[0].1, "head_arg")
                        .unwrap()
                        .into_array_value();
                    assert_ne!(val.get_type().len(), 0);
                    let result = builder.build_extract_value(val, 0, "head_res").unwrap();
                    let result_val = builder
                        .build_alloca(result.get_type(), "head_res_val")
                        .unwrap();
                    builder.build_store(result_val, result).unwrap();
                    (result.get_type().as_basic_type_enum(), result_val)
                }
                Tail => {
                    let val = builder
                        .build_load(args[0].0.into_array_type(), args[0].1, "tail_arg")
                        .unwrap()
                        .into_array_value();
                    assert_ne!(val.get_type().len(), 0);
                    let result = builder
                        .build_extract_value(val, val.get_type().len() - 1, "tail_res")
                        .unwrap();
                    let result_val = builder
                        .build_alloca(result.get_type(), "tail_res_val")
                        .unwrap();
                    builder.build_store(result_val, result).unwrap();
                    (result.get_type().as_basic_type_enum(), result_val)
                }
                IsEmpty => {
                    // Op::IsEmpty is for Jib lists only, which for now we treat as fixed-length arrays. Since these are statically
                    // sized, we don't need to access anything and just return a constant.
                    let result_val = builder
                        .build_alloca(llvm_ctx.bool_type(), "list_is_empty")
                        .unwrap();
                    builder
                        .build_store(result_val, llvm_ctx.bool_type().const_zero())
                        .unwrap();
                    (llvm_ctx.bool_type().as_basic_type_enum(), result_val)
                }
                ZeroExtend(n) => {
                    let val = builder
                        .build_load(args[0].0.into_struct_type(), args[0].1, "zext_arg")
                        .unwrap()
                        .into_struct_value();

                    let result_val = builder
                        .build_alloca(val.get_type(), "zext_res_val")
                        .unwrap();
                    builder.build_store(result_val, val).unwrap();

                    let result_len_ptr = builder
                        .build_struct_gep(val.get_type(), result_val, 0, "result_len")
                        .unwrap();
                    builder
                        .build_store(
                            result_len_ptr,
                            llvm_ctx.i128_type().const_int(*n as u64, false),
                        )
                        .unwrap();
                    (val.get_type().as_basic_type_enum(), result_val)
                }
            }
        }
        // Compare the tag represented by ctor_a with the tag of whatever union struct results from exp
        Kind(ctor_a, exp) => {
            let (tag_a, _) = statics.get_tag(&ctor_a.base_name());
            let (union_b_ty, union_b_ptr) = j2l_exp(
                &(**exp),
                locals,
                statics,
                strings,
                builder,
                module,
                llvm_ctx,
            );
            let union_b_ty = union_b_ty.into_struct_type();
            let tag_b_ty = union_b_ty
                .get_field_type_at_index(0)
                .unwrap()
                .into_int_type();
            assert_eq!(tag_a.get_type(), tag_b_ty);

            let tag_b_ptr = builder
                .build_struct_gep(union_b_ty, union_b_ptr, 0, "union_b_ptr")
                .unwrap_or_else(|e| panic!("{e}"));
            let tag_b = builder
                .build_load(tag_b_ty, tag_b_ptr, "ctor_b_ld")
                .unwrap();
            let result = builder
                .build_int_compare(
                    llvm::IntPredicate::EQ,
                    tag_a,
                    tag_b.into_int_value(),
                    "kind_res",
                )
                .unwrap();
            let result_val = builder
                .build_alloca(llvm_ctx.bool_type(), "kind_res_val")
                .unwrap();
            builder.build_store(result_val, result).unwrap();
            (llvm_ctx.bool_type().as_basic_type_enum(), result_val)
        }

        // Return a pointer to the union represented by exp as the type represented by the ctor_a tag
        Unwrap(ctor_a, exp) => {
            let (tag_a, ty_a) = statics.get_tag(&ctor_a.base_name());
            let ty_a = llvm::BasicTypeEnum::try_from(ty_a).unwrap();
            let (union_b_ty, union_b_ptr) =
                j2l_exp(&**exp, locals, statics, strings, builder, module, llvm_ctx);
            let union_b_ty = union_b_ty.into_struct_type();
            let tag_b_ty = union_b_ty
                .get_field_type_at_index(0)
                .unwrap()
                .into_int_type();
            assert_eq!(tag_a.get_type(), tag_b_ty);

            let val_b_ptr = builder
                .build_struct_gep(union_b_ty, union_b_ptr, 1, "union_b_ptr")
                .unwrap_or_else(|e| panic!("{e}"));
            let val_b = builder.build_load(ty_a, val_b_ptr, "val_b_ld").unwrap();

            let result_val = builder
                .build_alloca(llvm_ctx.bool_type(), "res_val")
                .unwrap();
            builder.build_store(result_val, val_b).unwrap();
            (ty_a.as_basic_type_enum(), result_val)
        }

        Field(exp, field) => {
            let (struct_ty, struct_ptr) = j2l_exp(
                &(**exp),
                locals,
                statics,
                strings,
                builder,
                module,
                llvm_ctx,
            );
            let struct_ty = struct_ty.into_struct_type();
            let struct_name = struct_ty
                .get_name()
                .unwrap()
                .to_str()
                .unwrap_or_else(|e| panic!("{e}"));

            let field_idx = statics
                .types
                .struct_fields
                .get(struct_name)
                .unwrap()
                .get(&field.base_name())
                .unwrap()
                .clone();
            let field_ty = struct_ty.get_field_type_at_index(field_idx).unwrap();
            let field_ptr = builder
                .build_struct_gep(struct_ty, struct_ptr, field_idx, "field_gep")
                .unwrap();
            (field_ty.as_basic_type_enum(), field_ptr)
        }

        Struct(struct_id, exp_fields) => {
            let fields = exp_fields
                .iter()
                .map(|(field_id, field_exp)| {
                    (
                        field_id,
                        j2l_exp(
                            field_exp, locals, statics, strings, builder, module, llvm_ctx,
                        ),
                    )
                })
                .map(|(field_id, (field_ty, field_ptr))| {
                    let field_val = builder.build_load(field_ty, field_ptr, "field").unwrap();
                    (field_id.base_name(), field_ty, field_val)
                })
                .collect::<Vec<(isla::Name, llvm::BasicTypeEnum, llvm::BasicValueEnum)>>();

            let ty = statics.types.mappings.get(&struct_id.base_name()).unwrap();
            let ty = if let J2LType::Struct(ty) = ty {
                ty.clone()
            } else {
                panic!("")
            };
            let struct_name = ty
                .get_name()
                .unwrap()
                .to_str()
                .unwrap_or_else(|e| panic!("{e}"));
            let struct_fields = statics.types.struct_fields.get(struct_name).unwrap();

            let ptr = builder.build_alloca(ty, "const struct").unwrap();
            for (field_id, _, field_val) in fields.iter() {
                let field_idx = struct_fields.get(field_id).unwrap();

                let field_ptr = builder
                    .build_struct_gep(ty, ptr, *field_idx, "field")
                    .unwrap_or_else(|e| {
                        panic!(
                            "{e}: Bad field idx {field_idx} ({0}) for {ty}",
                            statics.name(field_id)
                        )
                    });
                builder.build_store(field_ptr, *field_val).unwrap();
            }
            (ty.as_basic_type_enum(), ptr)
        }
    }
}

/// Evaluate a Jib location to a pointer to an existing LLVM value
fn j2l_loc<'a>(
    loc: &isla::BlockLoc,
    locals: &mut J2LLocalContext<'a>,
    statics: &J2LStaticContext<'a>,
    builder: &llvm::Builder<'a>,
    llvm_ctx: &'a llvm::Context,
) -> (llvm::BasicTypeEnum<'a>, llvm::PointerValue<'a>) {
    use isla::BlockLoc::*;
    match loc {
        Id(id) => j2l_id(id, locals, statics),
        Field(loc, _, field) => {
            let (parent_ty, parent_ptr) = j2l_loc(loc, locals, statics, builder, llvm_ctx);
            let parent_ty = parent_ty.into_struct_type();
            let struct_name = parent_ty
                .get_name()
                .unwrap()
                .to_str()
                .unwrap_or_else(|e| panic!("{e}"));

            let fields = statics
                .types
                .struct_fields
                .get(struct_name)
                .unwrap_or_else(|| panic!("Unable to find fields for {struct_name}"));
            let field_idx = fields
                .get(&field.base_name())
                .unwrap_or_else(|| {
                    panic!("Unable to find field {field:?} ({0}) in mappings for {struct_name}: {fields:?}", statics.ssa_name(field))
                })
                .clone();

            let field_ty = parent_ty.get_field_type_at_index(field_idx).unwrap();
            let field_ptr = builder
                .build_struct_gep(parent_ty, parent_ptr, field_idx, "loc_field")
                .unwrap();
            (field_ty.as_basic_type_enum(), field_ptr)
        }
        Addr(loc) => j2l_loc(&(**loc), locals, statics, builder, llvm_ctx),
    }
}

/// Build LLVM instructions for the given Jib instruction
fn j2l_instr<'a>(
    instr: &isla::BlockInstr<isla::B64>,
    locals: &mut J2LLocalContext<'a>,
    statics: &J2LStaticContext<'a>,
    strings: &mut HashMap<String, llvm::GlobalValue<'a>>,
    builder: &llvm::Builder<'a>,
    module: &llvm::Module<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    use isla::BlockInstr::*;
    // eprintln!("{instr:?}");
    match instr {
        Decl(val, ty, _) => {
            let ty = j2l_ty(unssa_ty(ty), &statics.types, llvm_ctx);
            let ptr = if let llvm::BasicTypeEnum::ArrayType(_) = ty {
                builder
                    .build_alloca(
                        llvm_ctx.ptr_type(llvm::AddressSpace::default()),
                        &statics.ssa_name(val),
                    )
                    .unwrap()
            } else {
                builder.build_alloca(ty, &statics.ssa_name(val)).unwrap()
            };

            locals.new_val(val.base_name(), ty.as_basic_type_enum(), ptr);
        }
        Init(val, ty, exp, _) => {
            let (init_ty, init_ptr) =
                j2l_exp(exp, locals, statics, strings, builder, module, llvm_ctx);
            let ty = j2l_ty(unssa_ty(ty), &statics.types, llvm_ctx);
            let assign = llvm_assign_chk(ty, init_ty);

            // This handles the Jib case of a local vector being assigned to a global vector
            // The pointer to the global would've been copied to a local pointer instead of a local vector
            // being created. Then that local pointer would be passed to a vector primop which would modify
            // the global vector in place using that passed-in local pointer value that points to the global
            // vector.
            if let Assign::Nop = assign {
                // TODO: could we build some kind of assert here to guarantee the pointers have the same value?
                return;
            }

            if let Assign::Pointee = assign {
                let ptr = builder.build_alloca(ty, &statics.ssa_name(val)).unwrap();
                locals.new_val(val.base_name(), ty.as_basic_type_enum(), ptr);
                // memcpy has a granularity of bytes, and we need bits (we have no alignment guarantees right now). Instead we will load src_ptr into
                // a val and store into dst_ptr.
                let to_st = builder.build_load(init_ty, init_ptr, "to_str").unwrap();
                builder.build_store(ptr, to_st).unwrap();
            } else if let Assign::Pointer = assign {
                let ptr = builder
                    .build_alloca(llvm_ctx.ptr_type(llvm::AddressSpace::default()), "ptr")
                    .unwrap();
                locals.new_val(val.base_name(), ty.as_basic_type_enum(), ptr);

                builder.build_store(ptr, init_ptr).unwrap();
            }
        }
        Copy(loc, exp, _) => {
            let (src_ty, src_ptr) =
                j2l_exp(exp, locals, statics, strings, builder, module, llvm_ctx);
            let (dst_ty, dst_ptr) = j2l_loc(loc, locals, statics, builder, llvm_ctx);
            let assign = llvm_assign_chk(dst_ty, src_ty);

            match assign {
                Assign::Pointee => {
                    // memcpy has a granularity of bytes, and we need bits. Instead we will load src_ptr into
                    // a val and store into dst_ptr.
                    let to_st = builder.build_load(src_ty, src_ptr, "to_str").unwrap();
                    builder.build_store(dst_ptr, to_st).unwrap();
                }
                Assign::Pointer => {
                    builder.build_store(dst_ptr, src_ptr).unwrap();
                }
                Assign::Nop => {
                    // This handles the Jib case of a local vector being assigned to a global vector
                    // The pointer to the global would've been copied to a local pointer instead of a local vector
                    // being created. Then that local pointer would be passed to a vector primop which would modify
                    // the global vector in place using that passed-in local pointer value that points to the global
                    // vector.
                    // TODO: could we build some kind of assert here to guarantee the pointers have the same value?
                }
            }
        }
        Call(loc, _is_ext, f, args, _) => {
            let mut args = args
                .iter()
                .map(|arg| j2l_exp(arg, locals, statics, strings, builder, module, llvm_ctx))
                .collect::<Vec<(llvm::BasicTypeEnum, llvm::PointerValue)>>();
            let (ret_ty, ret_ptr) = j2l_loc(loc, locals, statics, builder, llvm_ctx);

            if let Some((f, param_pointee_tys)) = statics.get_fn(f) {
                // Regular function call
                args.insert(0, (ret_ty, ret_ptr));

                // Type-check
                args.iter()
                    .map(|(ty, _)| ty)
                    .zip(param_pointee_tys.iter())
                    .for_each(|(arg_ty, param_pointee_ty)| {
                        let param_pointee_ty =
                            llvm::BasicTypeEnum::try_from(*param_pointee_ty).unwrap();
                        // assert_eq!(*arg_ty, param_pointee_ty);
                        llvm_call_arg_chk(param_pointee_ty, *arg_ty);
                    });

                let args = args
                    .into_iter()
                    .map(|(_, arg_ptr)| llvm::BasicMetadataValueEnum::from(arg_ptr))
                    .collect::<Vec<llvm::BasicMetadataValueEnum>>();

                // println!("Building call...");
                // println!("f = {f:?}");
                // println!("f.params = {0:?}", f.get_params());
                // println!("args = {args:?}");
                builder.build_call(f, &args, "call").unwrap();
                // println!("Call built.");
            } else if let Some((parent, tag, ty)) = statics.types.union_ctors.get(f) {
                // Construct a union from arg[0] at the location
                let (arg_ty, arg_ptr) = args[0];
                let ret_ty = ret_ty.into_struct_type();
                // Sanity check
                let parent_ty =
                    if let J2LType::Union(ty) = statics.types.mappings.get(parent).unwrap() {
                        ty.clone()
                    } else {
                        panic!("Unknown union type {parent:?}");
                    };
                assert_eq!(ret_ty, parent_ty);

                // Copy the tag into place
                let ret_tag_ptr = builder
                    .build_struct_gep(ret_ty, ret_ptr, 0, "ret_tag_ptr")
                    .unwrap();
                // MISTAKE: memcpy has a granularity of bytes, and we need bits. Instead we will load src_ptr into
                // a val and store into dst_ptr.
                // builder
                //     .build_memcpy(
                //         ret_tag_ptr,
                //         1,
                //         tag.as_pointer_value(),
                //         1,
                //         tag.get_value_type().into_int_type().size_of(),
                //     )
                //     .unwrap();
                let to_st = builder
                    .build_load(
                        tag.get_value_type().into_int_type(),
                        tag.as_pointer_value(),
                        "to_st",
                    )
                    .unwrap();
                builder.build_store(ret_tag_ptr, to_st).unwrap();

                // Copy the value into place
                let ret_val_ptr = builder
                    .build_struct_gep(ret_ty, ret_ptr, 1, "ret_val_ptr")
                    .unwrap();
                // MISTAKE: memcpy has a granularity of bytes, and we need bits. Instead we will load src_ptr into
                // a val and store into dst_ptr.
                // builder
                //     .build_memcpy(
                //         ret_val_ptr,
                //         1,
                //         arg_ptr,
                //         1,
                //         llvm_ctx
                //             .i64_type()
                //             .const_int(llvm_ty_size(arg_ty) as u64, false),
                //     )
                //     .unwrap();
                let to_st = builder.build_load(arg_ty, arg_ptr, "to_st").unwrap();
                builder.build_store(ret_val_ptr, to_st).unwrap();
            } else if *f == isla::ABSTRACT_PRIMOP {
                // This is a call to an abstract function, all of whom return unit in the Armv9.4a spec
                // We won't do anything here except copy our LLVM-equivalent to unit over to the return loc
                let void = llvm_ctx.bool_type().const_int(0, false);
                builder.build_store(ret_ptr, void).unwrap();
            } else {
                panic!("Call to unknown function {0}", statics.name(f));
            }
        }
        // From isla-lib's comments:
        // "The idea beind the Monomorphize operation is it takes a bitvector identifier, and if that identifer has a
        // symbolic value, then it uses the SMT solver to find all the possible values for that bitvector and case splits
        // (i.e. forks) on them. This allows us to guarantee that certain bitvectors are non-symbolic, at the cost of
        // increasing the number of paths."
        // I think in our case, this can just be treated as an identity function. The only uses in Armv9p4a are in
        // zsail_mem_write and zsail_mem_write which definitely won't be executed
        Monomorphize(..) => eprintln!("Monomorphize!"),
        // These are inserted at the same time as the "mono" instructions in regular isla but for all other extern functions.
        PrimopUnary(loc, primop, arg, _) => {
            let (arg_ty, arg_ptr) =
                j2l_exp(arg, locals, statics, strings, builder, module, llvm_ctx);
            let (ret_ty, ret_ptr) = j2l_loc(loc, locals, statics, builder, llvm_ctx);

            let unary_variations = statics.primops.unary.get(primop).unwrap();
            let arg_tys = [ret_ty, arg_ty]
                .into_iter()
                .map(|arg_ty| llvm::BasicMetadataTypeEnum::from(arg_ty))
                .collect::<Vec<llvm::BasicMetadataTypeEnum>>();
            let arg_tys = absurd_hash_of_function_params(&arg_tys);
            let (name, f, param_pointee_tys) = unary_variations
                .get(&arg_tys)
                .unwrap_or_else(|| panic!("No unary variation {arg_tys}"));
            // println!("PrimopUnary: {name}");

            // Type-check
            llvm_call_arg_chk(
                llvm::BasicTypeEnum::try_from(param_pointee_tys[0]).unwrap(),
                ret_ty,
            );
            llvm_call_arg_chk(
                llvm::BasicTypeEnum::try_from(param_pointee_tys[1]).unwrap(),
                arg_ty,
            );

            let args = [ret_ptr, arg_ptr]
                .into_iter()
                .map(|arg_ptr| llvm::BasicMetadataValueEnum::from(arg_ptr))
                .collect::<Vec<llvm::BasicMetadataValueEnum>>();

            // println!("Building unary primop call...");
            // println!("f = {f:?}");
            // println!("f.params = {0:?}", f.get_params());
            // println!("args = {args:?}");
            builder.build_call(*f, &args, "call").unwrap();
            // println!("Call built.");
        }
        PrimopBinary(loc, primop, lhs, rhs, _) => {
            let (lhs_ty, lhs_ptr) =
                j2l_exp(lhs, locals, statics, strings, builder, module, llvm_ctx);
            let (rhs_ty, rhs_ptr) =
                j2l_exp(rhs, locals, statics, strings, builder, module, llvm_ctx);
            let (ret_ty, ret_ptr) = j2l_loc(loc, locals, statics, builder, llvm_ctx);

            let binary_variations = statics.primops.binary.get(primop).unwrap();
            let arg_tys = [ret_ty, lhs_ty, rhs_ty]
                .into_iter()
                .map(|arg_ty| llvm::BasicMetadataTypeEnum::from(arg_ty))
                .collect::<Vec<llvm::BasicMetadataTypeEnum>>();
            let arg_tys = absurd_hash_of_function_params(&arg_tys);
            let (name, f, param_pointee_tys) = binary_variations
                .get(&arg_tys)
                .unwrap_or_else(|| panic!("No binary variation {arg_tys}"));

            // Type-check
            llvm_call_arg_chk(
                llvm::BasicTypeEnum::try_from(param_pointee_tys[0]).unwrap(),
                ret_ty,
            );
            llvm_call_arg_chk(
                llvm::BasicTypeEnum::try_from(param_pointee_tys[1]).unwrap(),
                lhs_ty,
            );
            llvm_call_arg_chk(
                llvm::BasicTypeEnum::try_from(param_pointee_tys[2]).unwrap(),
                rhs_ty,
            );

            let args = [ret_ptr, lhs_ptr, rhs_ptr]
                .into_iter()
                .map(|arg_ptr| llvm::BasicMetadataValueEnum::from(arg_ptr))
                .collect::<Vec<llvm::BasicMetadataValueEnum>>();

            // println!("Building binary primop call...");
            // println!("f = {f:?}");
            // println!("f.params = {0:?}", f.get_params());
            // println!("args = {args:?}");
            builder.build_call(*f, &args, "call").unwrap();
            // println!("Call built.");
        }
        PrimopVariadic(loc, primop, args, _) => {
            let mut args = args
                .iter()
                .map(|arg| j2l_exp(arg, locals, statics, strings, builder, module, llvm_ctx))
                .collect::<Vec<(llvm::BasicTypeEnum, llvm::PointerValue)>>();
            let (ret_ty, ret_ptr) = j2l_loc(loc, locals, statics, builder, llvm_ctx);
            args.insert(0, (ret_ty, ret_ptr));

            let variadic_variations = statics.primops.variadic.get(primop).unwrap();
            let arg_tys = args
                .iter()
                .map(|(arg_ty, _)| llvm::BasicMetadataTypeEnum::from(*arg_ty))
                .collect::<Vec<llvm::BasicMetadataTypeEnum>>();
            let arg_tys = absurd_hash_of_function_params(&arg_tys);
            let (name, f, param_pointee_tys) = variadic_variations
                .get(&arg_tys)
                .unwrap_or_else(|| panic!("No variadic variation {arg_tys}"));
            // println!("PrimopVariadic: {name}");

            // Type-check
            args.iter()
                .map(|(ty, _)| ty)
                .zip(param_pointee_tys.iter())
                .enumerate()
                .for_each(|(i, (arg_ty, param_pointee_ty))| {
                    // println!("call arg chk {i}");
                    let param_pointee_ty =
                        llvm::BasicTypeEnum::try_from(*param_pointee_ty).unwrap();
                    llvm_call_arg_chk(param_pointee_ty, *arg_ty);
                });

            let args = args
                .into_iter()
                .map(|(_, arg_ptr)| llvm::BasicMetadataValueEnum::from(arg_ptr))
                .collect::<Vec<llvm::BasicMetadataValueEnum>>();

            // println!("Building variadic primop call...");
            // println!("f = {f:?}");
            // println!("f.params = {0:?}", f.get_params());
            // println!("args = {args:?}");
            builder.build_call(*f, &args, "call").unwrap();
            // println!("Call built.");
        }
    }
}

/// Build LLVM instructions for the block's terminator, creating new LLVM basic blocks if neccessary
fn j2l_terminator<'a>(
    terminator: &isla::Terminator,
    block_idx: NodeIndex,
    llvm_block: llvm::BasicBlock<'a>,
    jib_cfg: &isla::CFG<isla::B64>,
    llvm_cfg: &mut HashMap<NodeIndex, llvm::BasicBlock<'a>>,
    locals: &mut J2LLocalContext<'a>,
    statics: &J2LStaticContext<'a>,
    strings: &mut HashMap<String, llvm::GlobalValue<'a>>,
    builder: &llvm::Builder<'a>,
    module: &llvm::Module<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    use isla::Terminator::*;
    match terminator {
        Goto(label) => {
            let mut outgoing_blocks = jib_cfg
                .graph
                .neighbors_directed(block_idx, Direction::Outgoing);
            let dest_block_idx = outgoing_blocks.next().unwrap();
            assert!(outgoing_blocks.next().is_none());

            let destination_block = llvm_cfg.entry(dest_block_idx).or_insert_with(|| {
                llvm_ctx.insert_basic_block_after(llvm_block, &format!("{label}"))
            });

            builder
                .build_unconditional_branch(*destination_block)
                .unwrap();
        }
        Jump(cond, t_target, _) => {
            let (cmp_ty, cmp_ptr) =
                j2l_exp(cond, locals, statics, strings, builder, module, llvm_ctx);
            let cmp_ty = cmp_ty.into_int_type();
            assert_eq!(cmp_ty, llvm_ctx.bool_type());
            let comparison = builder
                .build_load(cmp_ty, cmp_ptr, "comparison")
                .unwrap()
                .into_int_value();

            let mut outgoing_blocks = jib_cfg
                .graph
                .neighbors_directed(block_idx, Direction::Outgoing)
                .detach();
            let (out_edge_a, out_node_a) = outgoing_blocks.next(&jib_cfg.graph).unwrap();
            let (out_edge_b, out_node_b) = outgoing_blocks.next(&jib_cfg.graph).unwrap();
            assert!(outgoing_blocks.next(&jib_cfg.graph).is_none());
            // Disambiguate which outgoing (edge, node) pair is the "then" case versus the "else" case
            let ((t_block_idx, t_edge_idx), (f_block_idx, f_edge_idx)) =
                if let (_, isla::Edge::Jump(true)) = jib_cfg.graph[out_edge_a] {
                    ((out_node_a, out_edge_a), (out_node_b, out_edge_b))
                } else {
                    ((out_node_b, out_edge_b), (out_node_a, out_edge_a))
                };
            assert!(matches!(
                jib_cfg.graph[t_edge_idx],
                (_, isla::Edge::Jump(true))
            ));
            assert!(matches!(
                jib_cfg.graph[f_edge_idx],
                (_, isla::Edge::Jump(false))
            ));

            let then_block = llvm_cfg
                .entry(t_block_idx)
                .or_insert_with(|| {
                    let name = format!("{t_target}");
                    llvm_ctx.insert_basic_block_after(llvm_block, &name)
                })
                .clone();
            let else_block = llvm_cfg
                .entry(f_block_idx)
                .or_insert_with(|| {
                    let name = format!("{0}", jib_cfg.graph[f_block_idx].label.unwrap());
                    llvm_ctx.insert_basic_block_after(llvm_block, &name)
                })
                .clone();

            builder
                .build_conditional_branch(comparison, then_block, else_block)
                .unwrap();
        }
        End => {
            builder.build_return(None).unwrap();
        }
        Exit(cause, _) => match cause {
            isla::ExitCause::AssertionFailure => {
                builder.build_unreachable().unwrap();
            }
            isla::ExitCause::MatchFailure => {
                builder.build_unreachable().unwrap();
            }
            isla::ExitCause::Explicit => {
                builder.build_return(None).unwrap();
            }
        },
        Arbitrary => {
            // Poison the return value if one exists
            if let Some((ret_ty, ret_ptr)) = locals.args.get(&isla::RETURN) {
                let ret_val = match ret_ty {
                    llvm::BasicTypeEnum::ArrayType(ty) => ty.get_poison().as_basic_value_enum(),
                    llvm::BasicTypeEnum::IntType(ty) => ty.get_poison().as_basic_value_enum(),
                    llvm::BasicTypeEnum::FloatType(ty) => ty.get_poison().as_basic_value_enum(),
                    llvm::BasicTypeEnum::PointerType(ty) => ty.get_poison().as_basic_value_enum(),
                    llvm::BasicTypeEnum::StructType(ty) => ty.get_poison().as_basic_value_enum(),
                    llvm::BasicTypeEnum::VectorType(ty) => ty.get_poison().as_basic_value_enum(),
                };
                builder.build_store(*ret_ptr, ret_val).unwrap();
            }
            builder.build_return(None).unwrap();
        }
        Continue => unreachable!("Cfg::label_every_block eliminates these"),
        MultiJump(jump_tree) => {
            unimplemented!("MultiJump terminators {jump_tree:?}")
        }
    }
}

/// Build an LLVM phi-node equivalent to the given Jib phi-node
fn j2l_phi_node<'a>(
    outgoing: &isla::SSAName,
    incoming: &[isla::SSAName],
    locals: &mut J2LLocalContext<'a>,
    statics: &J2LStaticContext<'a>,
    builder: &llvm::Builder<'a>,
    _llvm_ctx: &'a llvm::Context,
) {
    // I don't really know what this case means...
    if incoming.is_empty() {
        return;
    }
    let (out_ty, _) = j2l_id(&incoming[0], locals, statics);
    let llvm_phi = builder
        .build_phi(llvm::BasicTypeEnum::try_from(out_ty).unwrap(), "phi_node")
        .unwrap();
    let mut llvm_incoming = Vec::with_capacity(incoming.len());
    for id in incoming {
        let (val_ty, val_ptr) = j2l_id(id, locals, statics);
        assert_eq!(out_ty, val_ty);
        let parent = val_ptr
            .as_instruction()
            .unwrap_or_else(|| panic!("No instruction for {}", statics.ssa_name(id)))
            .get_parent()
            .unwrap_or_else(|| panic!("No parent for {}", statics.ssa_name(id)));
        llvm_incoming.push((Box::new(val_ptr), parent));
    }
    let incoming = llvm_incoming
        .iter()
        .map(|(v, p)| (v.as_ref() as &dyn llvm::BasicValue, p.clone()))
        .collect::<Vec<(&dyn llvm::BasicValue, llvm::BasicBlock)>>();

    llvm_phi.add_incoming(incoming.as_slice());
    let outgoing_val = builder
        .build_alloca(
            llvm::BasicTypeEnum::try_from(out_ty).unwrap(),
            "phi_res_val",
        )
        .unwrap();
    locals.new_val(outgoing.base_name(), out_ty, outgoing_val);
}

/// Build the LLVM function for the given Jib function
fn j2l_function<'a>(
    id: isla::Name,
    args: Vec<(isla::Name, &isla::Ty<isla::Name>)>,
    ret_ty: &isla::Ty<isla::Name>,
    instrs: Vec<isla::Instr<isla::Name, isla::B64>>,
    statics: &J2LStaticContext<'a>,
    strings: &mut HashMap<String, llvm::GlobalValue<'a>>,
    builder: &llvm::Builder<'a>,
    module: &llvm::Module<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    // eprintln!("Translating {}", statics.name(&id));
    // eprintln!("Args given: {args:?}");
    let (f, _) = statics.get_fn(&id).unwrap();
    // eprintln!("func: {f:?}");
    // eprintln!("params: {0:?}", f.get_params());
    let args = [&[(isla::RETURN, ret_ty)], args.as_slice()].concat();
    // eprintln!("Args iterd: {args:?}");
    let mut j2l_args = BTreeMap::new();
    for ((arg_id, arg_ty), param) in args.iter().zip(f.get_param_iter()) {
        let arg_ty = j2l_ty((*arg_ty).clone(), &statics.types, llvm_ctx);
        let prev = j2l_args.insert(arg_id.clone(), (arg_ty, param.into_pointer_value()));
        assert_eq!(prev, None);
    }

    // eprintln!("Args final {j2l_args:?}");
    let mut locals = J2LLocalContext {
        args: j2l_args,
        locals: HashMap::new(),
    };
    // eprintln!("Initial local context: {locals:?}");

    let labeled = isla::prune_labels(isla::label_instrs(instrs));
    let mut jib_cfg = isla::CFG::new(&labeled);
    jib_cfg.ssa();
    jib_cfg.label_every_block();

    // jib_cfg
    //     .dot(&mut std::io::stderr(), &statics.symtab)
    //     .unwrap();

    let mut llvm_cfg: HashMap<NodeIndex, llvm::BasicBlock> = HashMap::new();
    let entry_block = llvm_ctx.append_basic_block(f, "entry");
    llvm_cfg.insert(jib_cfg.root, entry_block);

    let mut dfs = Dfs::new(&jib_cfg.graph, jib_cfg.root);
    while let Some(block_idx) = dfs.next(&jib_cfg.graph) {
        let jib_block = &jib_cfg.graph[block_idx];
        let llvm_block = llvm_cfg.get(&block_idx).unwrap().clone();
        builder.position_at_end(llvm_block);

        // Do phi nodes not matter because we make everything a stack variable that is accessed via ld/st? The
        // only way these could matter is if we did control-flow inside the handling of an expression. This also
        // means that "SSAName" doesn't matter, just base name.
        // for phi in &jib_block.phis {
        //     j2l_phi_node(&phi.0, &phi.1, &mut locals, statics, builder, llvm_ctx);
        // }

        for instr in &jib_block.instrs {
            j2l_instr(
                instr,
                &mut locals,
                statics,
                strings,
                builder,
                module,
                llvm_ctx,
            );
        }

        j2l_terminator(
            &jib_block.terminator,
            block_idx,
            llvm_block,
            &jib_cfg,
            &mut llvm_cfg,
            &mut locals,
            statics,
            strings,
            builder,
            module,
            llvm_ctx,
        );
    }
}

fn isla_isa_config() -> isla::ISAConfig<isla::B64> {
    isla::ISAConfig {
        // These are actually used by initialize_architecture
        const_primops: HashMap::new(),
        probes: HashSet::new(),
        probe_functions: HashSet::new(),
        trace_functions: HashSet::new(),
        reset_registers: Vec::new(),
        reset_constraints: Vec::new(),
        function_assumptions: Vec::new(),
        default_registers: HashMap::new(),
        relaxed_registers: HashSet::new(),
        // These appear to be unused for our purposes
        pc: isla::RETURN, // symtab.get(&zencode::encode("PC")).unwrap(),
        translation_function: None,
        register_event_sets: HashMap::new(),
        assembler: isla::Tool {
            executable: PathBuf::new(),
            options: Vec::new(),
        },
        objdump: isla::Tool {
            executable: PathBuf::new(),
            options: Vec::new(),
        },
        nm: isla::Tool {
            executable: PathBuf::new(),
            options: Vec::new(),
        },
        linker: isla::Tool {
            executable: PathBuf::new(),
            options: Vec::new(),
        },
        page_table_base: 0,
        page_size: 0,
        s2_page_table_base: 0,
        s2_page_size: 0,
        default_page_table_setup: String::new(),
        thread_base: 0,
        thread_top: 0,
        thread_stride: 0,
        symbolic_addr_base: 0,
        symbolic_addr_top: 0,
        symbolic_addr_stride: 0,
        register_renames: HashMap::new(),
        ignored_registers: HashSet::new(),
        in_program_order: HashSet::new(),
        default_sizeof: 0,
        zero_announce_exit: false,
    }
}

fn main() {
    let contents = {
        let mut contents = String::new();
        let mut input_ir = std::fs::File::open("isla-snapshots/armv9p4.ir").unwrap();
        if let Ok(num_bytes) = input_ir.read_to_string(&mut contents) {
            println!("Read {num_bytes} bytes");
        } else {
            panic!("Failed to read...");
        }
        contents
    };

    let (symtab, mut parsed_ir) = {
        let mut symtab = isla::Symtab::new();
        let parsed_ir = isla::IrParser::new()
            .parse(&mut symtab, isla::new_ir_lexer(&contents))
            .unwrap();
        (symtab, parsed_ir)
    };

    // Need to clone the arch because the lifetime persists in jib_spec and we need to iterate over the arch later
    let type_info = isla::IRTypeInfo::new(&parsed_ir);
    let isa_config = isla_isa_config();
    let jib_spec = isla::initialize_architecture(
        &mut parsed_ir,
        symtab,
        type_info,
        &isa_config,
        isla::AssertionMode::Optimistic,
        true,
    );
    let parsed_fns = jib_spec.shared_state.functions.clone();
    println!("[*] Jib spec loaded.");

    let llvm_ctx = llvm::Context::create();
    let builder = llvm_ctx.create_builder();
    let module = llvm_ctx.create_module("armv9p4");

    let statics = J2LStaticContext::new(&jib_spec, &module, &llvm_ctx);
    println!("[*] Jib -> LLVM statics done.");
    let mut strings = HashMap::new();
    // let mut let_bindings = HashSet::new();
    for (i, (id, (args, ret_ty, instrs))) in parsed_fns.iter().enumerate() {
        println!("***************");
        println!(
            "*** {0}/{1}: {2} ***",
            i + 1,
            parsed_fns.len(),
            statics.name(id)
        );
        println!("***************");

        let name = statics.name(id);
        if name.starts_with("z__Decode") || name.contains("SysReg") {
            // println!("Skipping (for dev speed) {name}");
            continue;
        }

        let (args, ret_ty, instrs) = jib_spec.shared_state.functions.get(id).unwrap();
        j2l_function(
            id.clone(),
            args.clone(),
            ret_ty,
            instrs.to_vec(),
            &statics,
            &mut strings,
            &builder,
            &module,
            &llvm_ctx,
        );
    }

    module.write_bitcode_to_path(std::path::Path::new("data/armv9p4a.bc"));
}
