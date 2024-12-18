mod llvm {
    pub use inkwell::basic_block::BasicBlock;
    pub use inkwell::builder::Builder;
    pub use inkwell::context::Context;
    pub use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    pub use inkwell::intrinsics::Intrinsic;
    pub use inkwell::module::{Linkage, Module};
    pub use inkwell::passes::PassBuilderOptions;
    pub use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };
    pub use inkwell::types::{
        AnyType, AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType,
        IntType, StructType,
    };
    pub use inkwell::values::{
        ArrayValue, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, GlobalValue,
        IntValue, PointerValue, StructValue,
    };
    pub use inkwell::{
        AddressSpace, FloatPredicate, GlobalVisibility, IntPredicate, OptimizationLevel,
        ThreadLocalMode,
    };
}
use llvm::BasicType;
use llvm::BasicValue;

fn get_native_target_machine() -> llvm::TargetMachine {
    llvm::Target::initialize_native(&llvm::InitializationConfig::default())
        .expect("Failed to initialize native target");
    let target_triple = llvm::TargetMachine::get_default_triple();
    let target = llvm::Target::from_triple(&target_triple).unwrap();
    target
        .create_target_machine(
            &target_triple,
            llvm::TargetMachine::get_host_cpu_name()
                .to_string()
                .as_str(),
            llvm::TargetMachine::get_host_cpu_features()
                .to_string()
                .as_str(),
            llvm::OptimizationLevel::None,
            llvm::RelocMode::Default,
            llvm::CodeModel::Default,
        )
        .unwrap()
}

fn main() {
    let context = llvm::Context::create();
    let module = llvm::Module::parse_bitcode_from_path("data/armv9p4a.bc", &context).unwrap();

    let entry_points = [
        "decode_add_addsub_imm_aarch64_instrs_integer_arithmetic_add_sub_immediate",
        "decode_subs_addsub_imm_aarch64_instrs_integer_arithmetic_add_sub_immediate",
        "decode_b_cond_aarch64_instrs_branch_conditional_cond",
        "decode_ldur_gen_aarch64_instrs_memory_single_general_immediate_signed_offset_normal",
    ];

    // Make patches
    // TODO
    // 1. Handwrite functions that ARM's pseudocde left open endeded:
    //  - PhysMemRead/Write
    //  - ImpDefInt/Bool/Bits (means we need real string support)
    // 2. Set certain registers as constants for the sake of optimizing around the minimum features I
    // want to support
    //  - Basically just A64 base
    // 3. Re-write the entry points' to take immediates for params instead of pointers (probably hardest cuz have
    // to re-write entire functions).
    // 4. Re-write pedantic pseudocode, e.g. AddWithCarry replaced with just add or intrinsics

    for glob in module.get_globals() {
        if glob.is_constant() {
            glob.set_linkage(llvm::Linkage::Internal);
        } else {
            let name = glob.get_name().to_str().unwrap();
            if !(name == "R0"
                || name == "R1"
                || name == "R2"
                || name == "R3"
                || name == "R4"
                || name == "R5"
                || name == "R6"
                || name == "R7"
                || name == "R8"
                || name == "R9"
                || name == "R10"
                || name == "R11"
                || name == "R12"
                || name == "R13"
                || name == "R14"
                || name == "R15"
                || name == "R16"
                || name == "R17"
                || name == "R18"
                || name == "R19"
                || name == "R20"
                || name == "R21"
                || name == "R22"
                || name == "R23"
                || name == "R24"
                || name == "R25"
                || name == "R26"
                || name == "R27"
                || name == "R28"
                || name == "R29"
                || name == "R30"
                || name == "PSTATE"
                || name == "SP_EL0"
                || name == "SP_EL1"
                || name == "SP_EL2"
                || name == "SP_EL3"
                || name == "_PC"
                || name == "zinq_mem_map")
            {
                glob.set_linkage(llvm::Linkage::Internal);
                let ty = glob.get_value_type();
                let init = match ty {
                    llvm::AnyTypeEnum::ArrayType(ty) => ty.const_zero().as_basic_value_enum(),
                    llvm::AnyTypeEnum::IntType(ty) => ty.const_zero().as_basic_value_enum(),
                    llvm::AnyTypeEnum::StructType(ty) => ty.const_zero().as_basic_value_enum(),
                    llvm::AnyTypeEnum::FloatType(ty) => ty.const_zero().as_basic_value_enum(),
                    other => unreachable!("{other:?}"),
                };
                glob.set_initializer(&init);
            }
        }
    }

    // Label all non-extern, non-entry-point functions as internal
    for f in module.get_functions() {
        // Re-name main. No idea if this matters but I'm superstitious
        if f.get_name().to_str().unwrap() == "main" {
            f.as_global_value().set_name("old_main");
        }

        if !entry_points.contains(&f.get_name().to_str().unwrap()) && f.count_basic_blocks() > 0 {
            f.set_linkage(llvm::Linkage::Internal);
        }
    }

    // Run dead global elimination to get rid of anything unused by the entry points
    let machine = get_native_target_machine();
    let options = llvm::PassBuilderOptions::create();
    // module.run_passes("default<O3>", &machine, options).unwrap();
    module.run_passes("globaldce", &machine, options).unwrap();

    module.write_bitcode_to_path(std::path::Path::new("data/armv9p4a_prepped.bc"));
}
