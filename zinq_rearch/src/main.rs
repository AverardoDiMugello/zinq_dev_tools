#![feature(generic_const_exprs)]

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use bitvec::prelude::*;

use capstone::{
    arch::{arm64::ArchMode, BuildsCapstone, BuildsCapstoneEndian},
    Capstone, Endian,
};

/// Re-export inkwell types from this module for use under the prefix llvm::
mod llvm {
    pub use inkwell::attributes::{Attribute, AttributeLoc};
    pub use inkwell::builder::Builder;
    pub use inkwell::context::Context;
    pub use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    pub use inkwell::intrinsics::Intrinsic;
    pub use inkwell::module::Module;
    pub use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };
    pub use inkwell::types::{
        AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType,
    };
    pub use inkwell::values::{
        BasicValue, BasicValueEnum, FunctionValue, GlobalValue, IntValue, StructValue,
    };
    pub use inkwell::{
        AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel, ThreadLocalMode,
    };
}

// mod arm;
// mod zinq;

#[repr(C)]
#[derive(Debug)]
struct Bv {
    len: u128,
    bits: u128,
}

impl Bv {
    fn new(len: u128, bits: u128) -> Self {
        Self { len, bits }
    }

    fn empty() -> Self {
        Self { len: 0, bits: 0 }
    }

    fn bit(b: bool) -> Self {
        Self {
            len: 1,
            bits: if b { 1 } else { 0 },
        }
    }

    fn bv64(bits: u64) -> Self {
        Self {
            len: 64,
            bits: bits as u128,
        }
    }
}

impl std::fmt::Display for Bv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bv {{ 0x{0:08x}, {1} }}", self.bits, self.len)
    }
}

#[repr(C)]
#[derive(Debug)]
struct ProcState {
    a: Bv,
    d: Bv,
    f: Bv,
    i: Bv,
    sp: Bv,
    z: Bv,
    allint: Bv,
    btype: Bv,
    c: Bv,
    dit: Bv,
    e: Bv,
    el: Bv,
    exlock: Bv,
    ge: Bv,
    il: Bv,
    it: Bv,
    j: Bv,
    m: Bv,
    n: Bv,
    pan: Bv,
    pm: Bv,
    ppend: Bv,
    q: Bv,
    sm: Bv,
    ss: Bv,
    ssbs: Bv,
    t: Bv,
    tco: Bv,
    uao: Bv,
    v: Bv,
    za: Bv,
    n_rw: Bv,
}

impl ProcState {
    fn new() -> Self {
        Self {
            a: Bv::bit(false),
            d: Bv::bit(false),
            f: Bv::bit(false),
            i: Bv::bit(false),
            sp: Bv::bit(false),
            z: Bv::bit(false),
            allint: Bv::bit(false),
            btype: Bv::new(2, 0),
            c: Bv::bit(false),
            dit: Bv::bit(false),
            e: Bv::bit(false),
            el: Bv::new(2, 0),
            exlock: Bv::bit(false),
            ge: Bv::new(4, 0),
            il: Bv::bit(false),
            it: Bv::new(8, 0),
            j: Bv::bit(false),
            m: Bv::new(5, 0),
            n: Bv::bit(false),
            pan: Bv::bit(false),
            pm: Bv::bit(false),
            ppend: Bv::bit(false),
            q: Bv::bit(false),
            sm: Bv::bit(false),
            ss: Bv::bit(false),
            ssbs: Bv::bit(false),
            t: Bv::bit(false),
            tco: Bv::bit(false),
            uao: Bv::bit(false),
            v: Bv::bit(false),
            za: Bv::bit(false),
            n_rw: Bv::bit(false),
        }
    }
}

type DecodeAddSubImmFn = unsafe extern "C" fn(
    *mut Bv,
    *const Bv,
    *const Bv,
    *const Bv,
    *const Bv,
    *const Bv,
    *const Bv,
    *const Bv,
);

type DecodeBCondImmFn = unsafe extern "C" fn(*mut Bv, *const Bv, *const Bv);

type DecodeLdurFn =
    unsafe extern "C" fn(*mut Bv, *const Bv, *const Bv, *const Bv, *const Bv, *const Bv);

fn disasm(raw: u32, pc: u64, cap: &Capstone) -> String {
    let bytes = raw.to_le_bytes();
    let disas = cap
        .disasm_all(&bytes, pc)
        .expect("Unable to disassemble instruction");
    let disas = disas
        .first()
        .and_then(|insn| Some(format!("{insn}")))
        .unwrap_or_else(|| {
            String::from("Instruction was disassembled but could not be unpacked from Capstone")
        });
    disas
}

fn init_capstone() -> Capstone {
    Capstone::new()
        .arm64()
        .mode(ArchMode::Arm)
        .endian(Endian::Little)
        .build()
        .expect("Unable to build Arm64 Capstone instance")
}

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

#[no_mangle]
extern "C" fn zinq_read_mem(ptr: *mut MemoryMap, addr: u64, bytes: u64) -> u64 {
    println!("zinq_read_mem!!!");
    let mem = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };

    let addr = addr as usize;
    match bytes {
        1 => mem.mem.get(addr..(addr + 1)),
        2 => mem.mem.get(addr..(addr + 2)),
        4 => mem.mem.get(addr..(addr + 4)),
        8 => mem.mem.get(addr..(addr + 8)),
        _ => unreachable!("{bytes}"),
    }
    .unwrap()
    .read_u64::<LittleEndian>()
    .unwrap()
}

#[no_mangle]
extern "C" fn zinq_write_mem(ptr: *mut MemoryMap, addr: u64, data: u64, data_len: u64) {
    println!("zinq_write_mem!!!");
    let mem = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };

    let addr = addr as usize;
    match data_len {
        1 => mem.mem.get_mut(addr..(addr + 1)),
        2 => mem.mem.get_mut(addr..(addr + 2)),
        4 => mem.mem.get_mut(addr..(addr + 4)),
        8 => mem.mem.get_mut(addr..(addr + 8)),
        _ => unreachable!("{data_len}"),
    }
    .unwrap()
    .write_u64::<LittleEndian>(data)
    .unwrap();
}

struct MemoryMap {
    mem: Vec<u8>,
}

impl MemoryMap {
    fn new(size: usize) -> Self {
        Self { mem: vec![0; size] }
    }

    // extern "C" fn zinq_read_mem(&mut self, addr: u64, bytes: u64) -> u64 {
    //     println!("zinq_read_mem!!!");
    //     let addr = addr as usize;
    //     match bytes {
    //         1 => self.mem.get(addr..(addr + 1)),
    //         2 => self.mem.get(addr..(addr + 2)),
    //         4 => self.mem.get(addr..(addr + 4)),
    //         8 => self.mem.get(addr..(addr + 8)),
    //         _ => unreachable!("{bytes}"),
    //     }
    //     .unwrap()
    //     .read_u64::<LittleEndian>()
    //     .unwrap()
    // }

    // extern "C" fn zinq_write_mem(&mut self, addr: u64, data: u64, data_len: u64) {
    //     println!("zinq_write_mem!!!");
    //     let addr = addr as usize;
    //     match data_len {
    //         1 => self.mem.get_mut(addr..(addr + 1)),
    //         2 => self.mem.get_mut(addr..(addr + 2)),
    //         4 => self.mem.get_mut(addr..(addr + 4)),
    //         8 => self.mem.get_mut(addr..(addr + 8)),
    //         _ => unreachable!("{data_len}"),
    //     }
    //     .unwrap()
    //     .write_u64::<LittleEndian>(data)
    //     .unwrap();
    // }
}

fn main() {
    let cap = init_capstone();
    // Input
    /*
     * x1 = x0 + 1;
     * if x1 >= 2 {
     *      x1 = *(x2 + 4);
     * }
     * x1 = x1 + 1;
     */
    let code: [u8; 20] = [
        0x01, 0x04, 0x00, 0x91, // add x1, x0, #1
        0x3F, 0x08, 0x00, 0xF1, // cmp x1, #2
        0x4B, 0x00, 0x00, 0x54, // b.lt #16
        0x41, 0x40, 0x40, 0xF8, // ldur x1, [x2, #4]
        0x21, 0x04, 0x00, 0x91, // add x1, x1, #1
    ];

    // Initial VM
    let mut pc = Box::new(Bv::bv64(0x0));

    let mut r0 = Box::new(Bv::bv64(0x1));
    let mut r1 = Box::new(Bv::bv64(0xfeedbedddddddd00));
    let mut r2 = Box::new(Bv::bv64(0x0));
    let mut r3 = Box::new(Bv::bv64(0xdeadbeefffffff03));
    let mut r4 = Box::new(Bv::bv64(0xdeadbeefffffff04));
    let mut r5 = Box::new(Bv::bv64(0xdeadbeefffffff05));
    let mut r6 = Box::new(Bv::bv64(0xdeadbeefffffff06));
    let mut r7 = Box::new(Bv::bv64(0xdeadbeefffffff07));
    let mut r8 = Box::new(Bv::bv64(0xdeadbeefffffff08));
    let mut r9 = Box::new(Bv::bv64(0xdeadbeefffffff09));
    let mut r10 = Box::new(Bv::bv64(0xdeadbeefffffff10));
    let mut r11 = Box::new(Bv::bv64(0xdeadbeefffffff11));
    let mut r12 = Box::new(Bv::bv64(0xdeadbeefffffff12));
    let mut r13 = Box::new(Bv::bv64(0xdeadbeefffffff13));
    let mut r14 = Box::new(Bv::bv64(0xdeadbeefffffff14));
    let mut r15 = Box::new(Bv::bv64(0xdeadbeefffffff15));
    let mut r16 = Box::new(Bv::bv64(0xdeadbeefffffff16));
    let mut r17 = Box::new(Bv::bv64(0xdeadbeefffffff17));
    let mut r18 = Box::new(Bv::bv64(0xdeadbeefffffff18));
    let mut r19 = Box::new(Bv::bv64(0xdeadbeefffffff19));
    let mut r20 = Box::new(Bv::bv64(0xdeadbeefffffff20));
    let mut r21 = Box::new(Bv::bv64(0xdeadbeefffffff21));
    let mut r22 = Box::new(Bv::bv64(0xdeadbeefffffff22));
    let mut r23 = Box::new(Bv::bv64(0xdeadbeefffffff23));
    let mut r24 = Box::new(Bv::bv64(0xdeadbeefffffff24));
    let mut r25 = Box::new(Bv::bv64(0xdeadbeefffffff25));
    let mut r26 = Box::new(Bv::bv64(0xdeadbeefffffff26));
    let mut r27 = Box::new(Bv::bv64(0xdeadbeefffffff27));
    let mut r28 = Box::new(Bv::bv64(0xdeadbeefffffff28));
    let mut r29 = Box::new(Bv::bv64(0xdeadbeefffffff29));
    let mut r30 = Box::new(Bv::bv64(0xdeadbeefffffff30));

    let mut pstate = Box::new(ProcState::new());
    let mut sp_el0 = Box::new(Bv::bv64(0xfedcba9876543210));
    let mut sp_el1 = Box::new(Bv::bv64(0xfedcba9876543211));
    let mut sp_el2 = Box::new(Bv::bv64(0xfedcba9876543212));
    let mut sp_el3 = Box::new(Bv::bv64(0xfedcba9876543213));

    println!("PC = {0}", *pc);
    println!("R0 = {0}", *r0);
    println!("R1 = {0}", *r1);
    println!("R2 = {0}", *r2);
    println!("R3 = {0}", *r3);
    println!("R4 = {0}", *r4);
    println!("R5 = {0}", *r5);
    println!("R6 = {0}", *r6);
    println!("R7 = {0}", *r7);
    println!("R8 = {0}", *r8);
    println!("R9 = {0}", *r9);
    println!("R10 = {0}", *r10);
    println!("R11 = {0}", *r11);
    println!("R12 = {0}", *r12);
    println!("R13 = {0}", *r13);
    println!("R14 = {0}", *r14);
    println!("R15 = {0}", *r15);
    println!("R16 = {0}", *r16);
    println!("R17 = {0}", *r17);
    println!("R18 = {0}", *r18);
    println!("R19 = {0}", *r19);
    println!("R20 = {0}", *r20);
    println!("R21 = {0}", *r21);
    println!("R22 = {0}", *r22);
    println!("R23 = {0}", *r23);
    println!("R24 = {0}", *r24);
    println!("R25 = {0}", *r25);
    println!("R26 = {0}", *r26);
    println!("R27 = {0}", *r27);
    println!("R28 = {0}", *r28);
    println!("R29 = {0}", *r29);
    println!("R30 = {0}", *r30);
    println!("PSTATE.N = {0}", pstate.n);
    println!("PSTATE.Z = {0}", pstate.z);
    println!("PSTATE.C = {0}", pstate.c);
    println!("PSTATE.V = {0}", pstate.v);
    println!("PSTATE = {0:?}", pstate);
    println!("SP_EL0 = {0}", *sp_el0);
    println!("SP_EL1 = {0}", *sp_el1);
    println!("SP_EL2 = {0}", *sp_el2);
    println!("SP_EL3 = {0}", *sp_el3);

    let mut mem = Box::new(MemoryMap::new(0x1000));
    (&mut mem.mem[0..8]).copy_from_slice(&0xeeeeeeeeeeeeeeeeu64.to_le_bytes());
    (&mut mem.mem[8..16]).copy_from_slice(&0xddddddddddddddddu64.to_le_bytes());

    // Translate
    let context = llvm::Context::create();
    let module = llvm::Module::parse_bitcode_from_path(
        std::path::Path::new("data/armv9p4a_prepped.bc"),
        &context,
    )
    .unwrap();

    println!("LLVM loaded.");

    // Map in global context
    let exec_engine = module
        .create_jit_execution_engine(llvm::OptimizationLevel::None)
        .unwrap();

    let ptr_to_pc = Box::into_raw(pc);
    exec_engine.add_global_mapping(
        module.get_global("_PC").as_ref().unwrap(),
        ptr_to_pc as usize,
    );
    // let ptr_to_r0 = (&mut *r0) as *mut u64;
    let ptr_to_r0 = Box::into_raw(r0);
    exec_engine.add_global_mapping(
        module.get_global("R0").as_ref().unwrap(),
        ptr_to_r0 as usize,
    );
    // let ptr_to_r1 = (&mut *r1) as *mut u64;
    let ptr_to_r1 = Box::into_raw(r1);
    exec_engine.add_global_mapping(
        module.get_global("R1").as_ref().unwrap(),
        ptr_to_r1 as usize,
    );
    let ptr_to_r2 = Box::into_raw(r2);
    exec_engine.add_global_mapping(
        module.get_global("R2").as_ref().unwrap(),
        ptr_to_r2 as usize,
    );
    let ptr_to_r3 = Box::into_raw(r3);
    exec_engine.add_global_mapping(
        module.get_global("R3").as_ref().unwrap(),
        ptr_to_r3 as usize,
    );
    let ptr_to_r4 = Box::into_raw(r4);
    exec_engine.add_global_mapping(
        module.get_global("R4").as_ref().unwrap(),
        ptr_to_r4 as usize,
    );
    let ptr_to_r5 = Box::into_raw(r5);
    exec_engine.add_global_mapping(
        module.get_global("R5").as_ref().unwrap(),
        ptr_to_r5 as usize,
    );
    let ptr_to_r6 = Box::into_raw(r6);
    exec_engine.add_global_mapping(
        module.get_global("R6").as_ref().unwrap(),
        ptr_to_r6 as usize,
    );
    let ptr_to_r7 = Box::into_raw(r7);
    exec_engine.add_global_mapping(
        module.get_global("R7").as_ref().unwrap(),
        ptr_to_r7 as usize,
    );
    let ptr_to_r8 = Box::into_raw(r8);
    exec_engine.add_global_mapping(
        module.get_global("R8").as_ref().unwrap(),
        ptr_to_r8 as usize,
    );
    let ptr_to_r9 = Box::into_raw(r9);
    exec_engine.add_global_mapping(
        module.get_global("R9").as_ref().unwrap(),
        ptr_to_r9 as usize,
    );
    // let ptr_to_r0 = (&mut *r0) as *mut u64;
    let ptr_to_r10 = Box::into_raw(r10);
    exec_engine.add_global_mapping(
        module.get_global("R10").as_ref().unwrap(),
        ptr_to_r10 as usize,
    );
    // let ptr_to_r11 = (&mut *r1) 1as *mut u64;
    let ptr_to_r11 = Box::into_raw(r11);
    exec_engine.add_global_mapping(
        module.get_global("R11").as_ref().unwrap(),
        ptr_to_r11 as usize,
    );
    let ptr_to_r12 = Box::into_raw(r12);
    exec_engine.add_global_mapping(
        module.get_global("R12").as_ref().unwrap(),
        ptr_to_r12 as usize,
    );
    let ptr_to_r13 = Box::into_raw(r13);
    exec_engine.add_global_mapping(
        module.get_global("R13").as_ref().unwrap(),
        ptr_to_r13 as usize,
    );
    let ptr_to_r14 = Box::into_raw(r14);
    exec_engine.add_global_mapping(
        module.get_global("R14").as_ref().unwrap(),
        ptr_to_r14 as usize,
    );
    let ptr_to_r15 = Box::into_raw(r15);
    exec_engine.add_global_mapping(
        module.get_global("R15").as_ref().unwrap(),
        ptr_to_r15 as usize,
    );
    let ptr_to_r16 = Box::into_raw(r16);
    exec_engine.add_global_mapping(
        module.get_global("R16").as_ref().unwrap(),
        ptr_to_r16 as usize,
    );
    let ptr_to_r17 = Box::into_raw(r17);
    exec_engine.add_global_mapping(
        module.get_global("R17").as_ref().unwrap(),
        ptr_to_r17 as usize,
    );
    let ptr_to_r18 = Box::into_raw(r18);
    exec_engine.add_global_mapping(
        module.get_global("R18").as_ref().unwrap(),
        ptr_to_r18 as usize,
    );
    let ptr_to_r19 = Box::into_raw(r19);
    exec_engine.add_global_mapping(
        module.get_global("R19").as_ref().unwrap(),
        ptr_to_r19 as usize,
    );
    // let ptr_to_r0 = (&mut *r0) as *mut u64;
    let ptr_to_r20 = Box::into_raw(r20);
    exec_engine.add_global_mapping(
        module.get_global("R20").as_ref().unwrap(),
        ptr_to_r20 as usize,
    );
    // let ptr_to_r21 = (&mut *r1) 2as *mut u64;
    let ptr_to_r21 = Box::into_raw(r21);
    exec_engine.add_global_mapping(
        module.get_global("R21").as_ref().unwrap(),
        ptr_to_r21 as usize,
    );
    let ptr_to_r22 = Box::into_raw(r22);
    exec_engine.add_global_mapping(
        module.get_global("R22").as_ref().unwrap(),
        ptr_to_r22 as usize,
    );
    let ptr_to_r23 = Box::into_raw(r23);
    exec_engine.add_global_mapping(
        module.get_global("R23").as_ref().unwrap(),
        ptr_to_r23 as usize,
    );
    let ptr_to_r24 = Box::into_raw(r24);
    exec_engine.add_global_mapping(
        module.get_global("R24").as_ref().unwrap(),
        ptr_to_r24 as usize,
    );
    let ptr_to_r25 = Box::into_raw(r25);
    exec_engine.add_global_mapping(
        module.get_global("R25").as_ref().unwrap(),
        ptr_to_r25 as usize,
    );
    let ptr_to_r26 = Box::into_raw(r26);
    exec_engine.add_global_mapping(
        module.get_global("R26").as_ref().unwrap(),
        ptr_to_r26 as usize,
    );
    let ptr_to_r27 = Box::into_raw(r27);
    exec_engine.add_global_mapping(
        module.get_global("R27").as_ref().unwrap(),
        ptr_to_r27 as usize,
    );
    let ptr_to_r28 = Box::into_raw(r28);
    exec_engine.add_global_mapping(
        module.get_global("R28").as_ref().unwrap(),
        ptr_to_r28 as usize,
    );
    // let ptr_to_r29 = (&mut *r29) as *mut u64;
    let ptr_to_r29 = Box::into_raw(r29);
    exec_engine.add_global_mapping(
        module.get_global("R29").as_ref().unwrap(),
        ptr_to_r29 as usize,
    );
    // let ptr_to_r30 = (&mut *r30) as *mut u64;
    let ptr_to_r30 = Box::into_raw(r30);
    exec_engine.add_global_mapping(
        module.get_global("R30").as_ref().unwrap(),
        ptr_to_r30 as usize,
    );
    // let ptr_to_pstate = (&mut *pstate) as *mut ProcState;
    let ptr_to_pstate = Box::into_raw(pstate);
    exec_engine.add_global_mapping(
        module.get_global("PSTATE").as_ref().unwrap(),
        ptr_to_pstate as usize,
    );
    let ptr_to_sp_el0 = Box::into_raw(sp_el0);
    exec_engine.add_global_mapping(
        module.get_global("SP_EL0").as_ref().unwrap(),
        ptr_to_sp_el0 as usize,
    );
    let ptr_to_sp_el1 = Box::into_raw(sp_el1);
    exec_engine.add_global_mapping(
        module.get_global("SP_EL1").as_ref().unwrap(),
        ptr_to_sp_el1 as usize,
    );
    let ptr_to_sp_el2 = Box::into_raw(sp_el2);
    exec_engine.add_global_mapping(
        module.get_global("SP_EL2").as_ref().unwrap(),
        ptr_to_sp_el2 as usize,
    );
    let ptr_to_sp_el3 = Box::into_raw(sp_el3);
    exec_engine.add_global_mapping(
        module.get_global("SP_EL3").as_ref().unwrap(),
        ptr_to_sp_el3 as usize,
    );

    let ptr_to_mem = Box::into_raw(mem);
    exec_engine.add_global_mapping(
        module.get_global("zinq_mem_map").as_ref().unwrap(),
        ptr_to_mem as usize,
    );
    exec_engine.add_global_mapping(
        module.get_function("zinq_read_mem").as_ref().unwrap(),
        zinq_read_mem as usize,
    );
    exec_engine.add_global_mapping(
        module.get_function("zinq_write_mem").as_ref().unwrap(),
        zinq_write_mem as usize,
    );

    let target_machine = get_native_target_machine();
    module.set_triple(&target_machine.get_triple());
    module.set_data_layout(&target_machine.get_target_data().get_data_layout());
    let path = std::path::Path::new(&"module_mcode");
    assert!(target_machine
        .write_to_file(&module, llvm::FileType::Object, &path)
        .is_ok());

    // Emulation loop
    let mut icount = 0;
    let stop_pc = 20;

    loop {
        let fake_pc = unsafe { (*ptr_to_pc).bits as usize };
        if fake_pc >= stop_pc {
            break;
        }

        // Fetch
        let insn = u32::from_le_bytes(code[fake_pc..(fake_pc + 4)].try_into().unwrap());
        println!("{icount}: {0}", disasm(insn, fake_pc as u64, &cap));

        // Decode/execute
        if insn == 0x01040091u32.swap_bytes() || insn == 0x21040091u32.swap_bytes() {
            let insn = insn.view_bits::<Lsb0>();
            let mut unit = Bv::empty();

            // Add and subs
            let rd = Bv::new(5, (&insn[0..5]).load::<u128>());
            let rn = Bv::new(5, (&insn[5..10]).load::<u128>());
            let imm12 = Bv::new(12, (&insn[10..22]).load::<u128>());
            let sh = Bv::bit(insn[22]);
            let s = Bv::bit(insn[29]);
            let op = Bv::bit(insn[30]);
            let sf = Bv::bit(insn[31]);

            // Note: These should probably not actually exist and just be passed in as constants. The entry point may need to have its params
            // updated to be values, but what's the best way to do that?
            let unit = (&mut unit) as *mut Bv;
            let rd = (&rd) as *const Bv;
            let rn = (&rn) as *const Bv;
            let imm12 = (&imm12) as *const Bv;
            let sh = (&sh) as *const Bv;
            let s = (&s) as *const Bv;
            let op = (&op) as *const Bv;
            let sf = (&sf) as *const Bv;

            let jit_compiled_fn: llvm::JitFunction<DecodeAddSubImmFn> = unsafe {
                exec_engine
                    .get_function(
                        "decode_add_addsub_imm_aarch64_instrs_integer_arithmetic_add_sub_immediate",
                    )
                    .unwrap()
            };

            println!("Host MCode of ADDS loaded.");

            unsafe { jit_compiled_fn.call(unit, rd, rn, imm12, sh, s, op, sf) };

            unsafe {
                (*ptr_to_pc).bits += 4;
            }
        } else if insn == 0x3F0800F1u32.swap_bytes() {
            let insn = insn.view_bits::<Lsb0>();
            let mut unit = Bv::empty();

            // Add and subs
            let rd = Bv::new(5, (&insn[0..5]).load::<u128>());
            let rn = Bv::new(5, (&insn[5..10]).load::<u128>());
            let imm12 = Bv::new(12, (&insn[10..22]).load::<u128>());
            let sh = Bv::bit(insn[22]);
            let s = Bv::bit(insn[29]);
            let op = Bv::bit(insn[30]);
            let sf = Bv::bit(insn[31]);

            // Note: These should probably not actually exist and just be passed in as constants. The entry point may need to have its params
            // updated to be values, but what's the best way to do that?
            let unit = (&mut unit) as *mut Bv;
            let rd = (&rd) as *const Bv;
            let rn = (&rn) as *const Bv;
            let imm12 = (&imm12) as *const Bv;
            let sh = (&sh) as *const Bv;
            let s = (&s) as *const Bv;
            let op = (&op) as *const Bv;
            let sf = (&sf) as *const Bv;

            let jit_compiled_fn: llvm::JitFunction<DecodeAddSubImmFn> = unsafe {
                exec_engine
                    .get_function(
                        "decode_subs_addsub_imm_aarch64_instrs_integer_arithmetic_add_sub_immediate",
                    )
                    .unwrap()
            };

            println!("Host MCode of SUBS loaded.");

            unsafe { jit_compiled_fn.call(unit, rd, rn, imm12, sh, s, op, sf) };

            unsafe {
                (*ptr_to_pc).bits += 4;
            }
        } else if insn == 0x4B000054u32.swap_bytes() {
            let insn = insn.view_bits::<Lsb0>();
            let mut unit = Bv::empty();

            // B.cond
            let imm19 = Bv::new(19, (&insn[5..24]).load::<u128>());
            let cond = Bv::new(4, (&insn[0..4]).load::<u128>());

            // Note: These should probably not actually exist and just be passed in as constants. The entry point may need to have its params
            // updated to be values, but what's the best way to do that?
            let unit = (&mut unit) as *mut Bv;
            let imm19 = (&imm19) as *const Bv;
            let cond = (&cond) as *const Bv;

            let jit_compiled_fn: llvm::JitFunction<DecodeBCondImmFn> = unsafe {
                exec_engine
                    .get_function("decode_b_cond_aarch64_instrs_branch_conditional_cond")
                    .unwrap()
            };

            println!("Host MCode of B.cond loaded.");

            unsafe { jit_compiled_fn.call(unit, cond, imm19) };

            // TODO: if branch taken
            unsafe {
                (*ptr_to_pc).bits += 4;
            }
        } else if insn == 0x414040F8u32.swap_bytes() {
            let insn = insn.view_bits::<Lsb0>();
            let mut unit = Bv::empty();

            let rt = Bv::new(5, (&insn[0..5]).load::<u128>());
            let rn = Bv::new(5, (&insn[5..10]).load::<u128>());
            let imm9 = Bv::new(12, (&insn[12..21]).load::<u128>());
            let opc = Bv::new(2, (&insn[22..24]).load::<u128>());
            let size = Bv::new(2, (&insn[30..32]).load::<u128>());

            let unit = (&mut unit) as *mut Bv;
            let rt = (&rt) as *const Bv;
            let rn = (&rn) as *const Bv;
            let imm9 = (&imm9) as *const Bv;
            let opc = (&opc) as *const Bv;
            let size = (&size) as *const Bv;

            let jit_compiled_fn: llvm::JitFunction<DecodeLdurFn> = unsafe {
                exec_engine
                    .get_function(
                        "decode_ldur_gen_aarch64_instrs_memory_single_general_immediate_signed_offset_normal",
                    )
                    .unwrap()
            };

            println!("Host MCode of LDUR loaded.");

            unsafe { jit_compiled_fn.call(unit, rt, rn, imm9, opc, size) };

            println!("Out of LDUR!");

            unsafe {
                (*ptr_to_pc).bits += 4;
            }
        }

        // Log
        println!("*** Start ICount {icount} ***");
        unsafe {
            println!("PC = {0}", *ptr_to_pc);

            println!("R0 = {0}", *ptr_to_r0);
            println!("R1 = {0}", *ptr_to_r1);
            println!("R2 = {0}", *ptr_to_r2);
            println!("R3 = {0}", *ptr_to_r3);
            println!("R4 = {0}", *ptr_to_r4);
            println!("R5 = {0}", *ptr_to_r5);
            println!("R6 = {0}", *ptr_to_r6);
            println!("R7 = {0}", *ptr_to_r7);
            println!("R8 = {0}", *ptr_to_r8);
            println!("R9 = {0}", *ptr_to_r9);
            println!("R10 = {0}", *ptr_to_r10);
            println!("R11 = {0}", *ptr_to_r11);
            println!("R12 = {0}", *ptr_to_r12);
            println!("R13 = {0}", *ptr_to_r13);
            println!("R14 = {0}", *ptr_to_r14);
            println!("R15 = {0}", *ptr_to_r15);
            println!("R16 = {0}", *ptr_to_r16);
            println!("R17 = {0}", *ptr_to_r17);
            println!("R18 = {0}", *ptr_to_r18);
            println!("R19 = {0}", *ptr_to_r19);
            println!("R20 = {0}", *ptr_to_r20);
            println!("R21 = {0}", *ptr_to_r21);
            println!("R22 = {0}", *ptr_to_r22);
            println!("R23 = {0}", *ptr_to_r23);
            println!("R24 = {0}", *ptr_to_r24);
            println!("R25 = {0}", *ptr_to_r25);
            println!("R26 = {0}", *ptr_to_r26);
            println!("R27 = {0}", *ptr_to_r27);
            println!("R28 = {0}", *ptr_to_r28);
            println!("R29 = {0}", *ptr_to_r29);
            println!("R30 = {0}", *ptr_to_r30);

            println!("PSTATE.N = {0}", (*ptr_to_pstate).n);
            println!("PSTATE.Z = {0}", (*ptr_to_pstate).z);
            println!("PSTATE.C = {0}", (*ptr_to_pstate).c);
            println!("PSTATE.V = {0}", (*ptr_to_pstate).v);
            println!("PSTATE = {0:?}", (*ptr_to_pstate));

            println!("SP_EL0 = {0}", *ptr_to_sp_el0);
            println!("SP_EL1 = {0}", *ptr_to_sp_el1);
            println!("SP_EL2 = {0}", *ptr_to_sp_el2);
            println!("SP_EL3 = {0}", *ptr_to_sp_el3);
        }
        println!("*** End ICount {icount} ***");

        icount += 1;

        if icount == 5 {
            panic!("Too much");
        }
    }
}
