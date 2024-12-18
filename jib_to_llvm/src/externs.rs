use super::llvm;
use super::llvm_deep_eq;
use super::J2LDeclaredTypes;

// declare void @pow2(ptr sret(i128), ptr byref(i128))
fn pow2<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder
        .build_load(llvm_ctx.i128_type(), arg_ptr, "arg")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_left_shift(llvm_ctx.i128_type().const_int(1, false), arg, "1 << arg")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @not(ptr sret(i1), ptr byref(i1))
fn not<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder
        .build_load(llvm_ctx.bool_type(), arg_ptr, "arg")
        .unwrap()
        .into_int_value();
    let ret = builder.build_not(arg, "!arg").unwrap();
    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @eq_bool(ptr sret(i1), ptr byref(i1), ptr byref(i1))
fn eq_bool<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.bool_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.bool_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();

    let ret = builder
        .build_int_compare(llvm::IntPredicate::EQ, lhs, rhs, "lhs == rhs")
        .unwrap();
    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @undefined_bitvector(ptr sret(%bv), ptr byref(i128))
fn undefined_bitvector<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let length_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let length = builder
        .build_load(llvm_ctx.i128_type(), length_ptr, "length")
        .unwrap()
        .into_int_value();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, length).unwrap();
    builder
        .build_store(ret_bits_ptr, llvm_ctx.i128_type().const_zero())
        .unwrap();
    builder.build_return(None).unwrap();
}

// declare void @undefined_bool(ptr sret(i1), ptr byref(%bv))
fn undefined_bool<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder
        .build_store(ret_ptr, llvm_ctx.bool_type().const_zero())
        .unwrap();
    builder.build_return(None).unwrap();
}

// declare void @undefined_int(ptr sret(i128), ptr byref(%bv))
fn undefined_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder
        .build_store(ret_ptr, llvm_ctx.i128_type().const_zero())
        .unwrap();
    builder.build_return(None).unwrap();
}

// declare void @not_bits(ptr sret(%bv), ptr byref(%bv))
fn not_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let ret_bits = builder.build_not(bv_bits, "!bv.bits").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, bv_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @lteq(ptr sret(i1), ptr byref(i128), ptr byref(i128))
fn lteq<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_compare(llvm::IntPredicate::SLE, lhs, rhs, "sle")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @slice(ptr sret(%bv), ptr byref(%bv), ptr byref(i128), ptr byref(i128))
fn slice<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let from_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let from = builder
        .build_load(llvm_ctx.i128_type(), from_ptr, "from")
        .unwrap()
        .into_int_value();

    let length_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let length = builder
        .build_load(llvm_ctx.i128_type(), length_ptr, "length")
        .unwrap()
        .into_int_value();

    let res_bits = builder
        .build_right_shift(bv_bits, from, false, "bits >> from")
        .unwrap();
    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, length, "1 << length")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << length) - 1")
        .unwrap();
    let res_bits = builder
        .build_and(res_bits, mask, "(bits >> from) & ((1 << length) - 1)")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, length).unwrap();
    builder.build_store(ret_bits_ptr, res_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @vector_subrange(ptr sret(%bv), ptr byref(%bv), ptr byref(i128), ptr byref(i128))
fn vector_subrange<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let high_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let high = builder
        .build_load(llvm_ctx.i128_type(), high_ptr, "high")
        .unwrap()
        .into_int_value();

    let low_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let low = builder
        .build_load(llvm_ctx.i128_type(), low_ptr, "low")
        .unwrap()
        .into_int_value();

    let res_bits = builder
        .build_right_shift(bv_bits, low, false, "bits >> low")
        .unwrap();
    let one = llvm_ctx.i128_type().const_int(1, false);
    let length = builder.build_int_sub(high, low, "high - low").unwrap();
    let length = builder
        .build_int_add(length, one, "high - low + 1")
        .unwrap();
    let mask = builder
        .build_left_shift(one, length, "1 << (high - low + 1)")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << (high - low + 1)) - 1")
        .unwrap();
    let res_bits = builder
        .build_and(
            res_bits,
            mask,
            "(bits >> low) & ((1 << (high - low + 1)) - 1)",
        )
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, length).unwrap();
    builder.build_store(ret_bits_ptr, res_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @shiftl(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn shiftl<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    // i128
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();

    let ret_len = builder
        .build_int_add(lhs_len, rhs, "lhs.len + rhs")
        .unwrap();
    let ret_bits = builder.build_left_shift(lhs_bits, rhs, "shiftl").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, ret_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @eq_bits(ptr sret(i1), ptr byref(%bv), ptr byref(%bv))
fn eq_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs_len_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 0, "&rhs.len")
        .unwrap();
    let rhs_len = builder
        .build_load(llvm_ctx.i128_type(), rhs_len_ptr, "rhs.len")
        .unwrap()
        .into_int_value();
    let rhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 1, "&rhs.bits")
        .unwrap();
    let rhs_bits = builder
        .build_load(llvm_ctx.i128_type(), rhs_bits_ptr, "rhs.bits")
        .unwrap()
        .into_int_value();

    let len_eq = builder
        .build_int_compare(llvm::IntPredicate::EQ, lhs_len, rhs_len, "len_eq")
        .unwrap();
    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, lhs_len, "1 << lhs.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << lhs.len) - 1")
        .unwrap();
    let lhs_bits = builder.build_and(lhs_bits, mask, "lhs.len & mask").unwrap();
    let rhs_bits = builder.build_and(rhs_bits, mask, "rhs.len & mask").unwrap();
    let bits_eq = builder
        .build_int_compare(llvm::IntPredicate::EQ, lhs_bits, rhs_bits, "bits_eq")
        .unwrap();
    let ret = builder.build_and(len_eq, bits_eq, "eq").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @neq_bits(ptr sret(i1), ptr byref(%bv), ptr byref(%bv))
fn neq_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs_len_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 0, "&rhs.len")
        .unwrap();
    let rhs_len = builder
        .build_load(llvm_ctx.i128_type(), rhs_len_ptr, "rhs.len")
        .unwrap()
        .into_int_value();
    let rhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 1, "&rhs.bits")
        .unwrap();
    let rhs_bits = builder
        .build_load(llvm_ctx.i128_type(), rhs_bits_ptr, "rhs.bits")
        .unwrap()
        .into_int_value();

    let len_eq = builder
        .build_int_compare(llvm::IntPredicate::EQ, lhs_len, rhs_len, "len_eq")
        .unwrap();
    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, lhs_len, "1 << lhs.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << lhs.len) - 1")
        .unwrap();
    let lhs_bits = builder.build_and(lhs_bits, mask, "lhs.len & mask").unwrap();
    let rhs_bits = builder.build_and(rhs_bits, mask, "rhs.len & mask").unwrap();
    let bits_eq = builder
        .build_int_compare(llvm::IntPredicate::NE, lhs_bits, rhs_bits, "bits_eq")
        .unwrap();
    let ret = builder.build_and(len_eq, bits_eq, "eq").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @lt(ptr sret(i1), ptr byref(i128), ptr byref(i128))
fn lt<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_compare(llvm::IntPredicate::SLT, lhs, rhs, "slt")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @gt(ptr sret(i1), ptr byref(i128), ptr byref(i128))
fn gt<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_compare(llvm::IntPredicate::SGT, lhs, rhs, "sgt")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sail_signed(ptr sret(i128), ptr byref(%bv))
fn sail_signed<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    // Zero everything above the length
    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, bv_len, "1 << bits.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << bits.len) - 1")
        .unwrap();
    let ret_bits = builder
        .build_and(bv_bits, mask, "bv.bits & ((1 << bits.len) - 1)")
        .unwrap();

    // Extend the sign bit to 128-bits
    let sign_pos = builder.build_int_sub(bv_len, one, "len - 1").unwrap();
    let sign_bit = builder
        .build_right_shift(bv_bits, sign_pos, false, "bv.bits >> (len - 1)")
        .unwrap();
    let sign_bit = builder
        .build_int_truncate(sign_bit, llvm_ctx.bool_type(), "sign")
        .unwrap();
    let sign_mask = builder
        .build_int_s_extend(sign_bit, llvm_ctx.i128_type(), "sext(sign, 128)")
        .unwrap();
    let sign_mask = builder
        .build_left_shift(sign_mask, bv_len, "sext(sign, 128) << bv.len")
        .unwrap();
    let ret_bits = builder
        .build_or(ret_bits, sign_mask, "ret_bits | sign_mask")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @get_slice_int(ptr sret(%bv), ptr byref(i128), ptr byref(i128), ptr byref(i128))
fn get_slice_int<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let length_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let length = builder
        .build_load(llvm_ctx.i128_type(), length_ptr, "length")
        .unwrap()
        .into_int_value();

    let n_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let n = builder
        .build_load(llvm_ctx.i128_type(), n_ptr, "n")
        .unwrap()
        .into_int_value();

    let from_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let from = builder
        .build_load(llvm_ctx.i128_type(), from_ptr, "from")
        .unwrap()
        .into_int_value();

    let res_bits = builder
        .build_right_shift(n, from, false, "n >> from")
        .unwrap();
    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, length, "1 << length")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << length) - 1")
        .unwrap();
    let res_bits = builder
        .build_and(res_bits, mask, "(n >> from) & ((1 << length) - 1)")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, length).unwrap();
    builder.build_store(ret_bits_ptr, res_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sail_truncate(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn sail_truncate<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let trunc_arg_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let trunc_arg = builder
        .build_load(llvm_ctx.i128_type(), trunc_arg_ptr, "trunc_arg")
        .unwrap()
        .into_int_value();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, trunc_arg).unwrap();
    builder.build_store(ret_bits_ptr, bv_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sub_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn sub_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder.build_int_sub(lhs, rhs, "sub").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @"%i->%i64"(ptr sret(i64), ptr byref(i128))
// In Isla, this is an extraction of the low 64-bits of the argument
fn i_to_i64<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder
        .build_load(llvm_ctx.i128_type(), arg_ptr, "arg")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_truncate(arg, llvm_ctx.i64_type(), "arg[0..64]")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @read_register_from_vector(ptr sret(i64), ptr byref(i64), ptr byref([31 x ptr]))
fn read_register_from_vector<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let n_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let n = builder
        .build_load(llvm_ctx.i64_type(), n_ptr, "n")
        .unwrap()
        .into_int_value();

    let reg_vec_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let reg_n_ptr = unsafe {
        // 0 (first array of pointers pointed at)
        // n (nth pointer in the array of pointer)
        builder.build_gep(
            llvm_ctx
                .ptr_type(llvm::AddressSpace::default())
                .array_type(31),
            reg_vec_ptr,
            &[llvm_ctx.i32_type().const_zero(), n],
            "ptr to reg_vec_ptr[n]",
        )
    }
    .unwrap();

    let reg_n = builder
        .build_load(
            llvm_ctx.ptr_type(llvm::AddressSpace::default()),
            reg_n_ptr,
            "reg_vec_ptr[n]",
        )
        .unwrap()
        .into_pointer_value();
    let val = builder
        .build_load(types.bitvec, reg_n, "read_from_reg")
        .unwrap()
        .into_struct_value();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, val).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @write_register_from_vector(ptr sret(i1), ptr byref(i64), ptr byref(i64), ptr byref([31 x ptr]))
fn write_register_from_vector<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let n_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let n = builder
        .build_load(llvm_ctx.i64_type(), n_ptr, "n")
        .unwrap()
        .into_int_value();

    let val_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let val = builder
        .build_load(types.bitvec, val_ptr, "val")
        .unwrap()
        .into_struct_value();

    let reg_vec_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let reg_n_ptr = unsafe {
        builder.build_gep(
            llvm_ctx
                .ptr_type(llvm::AddressSpace::default())
                .array_type(31),
            reg_vec_ptr,
            &[llvm_ctx.i32_type().const_zero(), n],
            "reg_vec_ptr[n]",
        )
    }
    .unwrap();

    let reg_n = builder
        .build_load(
            llvm_ctx.ptr_type(llvm::AddressSpace::default()),
            reg_n_ptr,
            "reg_vec_ptr[n]",
        )
        .unwrap()
        .into_pointer_value();
    builder.build_store(reg_n, val).unwrap();
    // No write to return
    builder.build_return(None).unwrap();
}

// declare void @length(ptr sret(i128), ptr byref(%bv))
fn length<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg_len_ptr = builder
        .build_struct_gep(types.bitvec, arg_ptr, 0, "arg_len_ptr")
        .unwrap();
    let arg_len = builder
        .build_load(llvm_ctx.i128_type(), arg_len_ptr, "arg_len")
        .unwrap()
        .into_int_value();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, arg_len).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @shiftr(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn shiftr<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    // i128
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();

    let ret_len = builder
        .build_int_sub(lhs_len, rhs, "lhs.len - rhs")
        .unwrap();
    let ret_bits = builder
        .build_right_shift(lhs_bits, rhs, false, "shiftr")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, ret_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @arith_shiftr(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn arith_shiftr<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();

    let ret_len = builder
        .build_int_sub(lhs_len, rhs, "lhs.len - rhs")
        .unwrap();
    let ret_bits = builder
        .build_right_shift(lhs_bits, rhs, true, "arith_shiftr")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, ret_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @zeros(ptr sret(%bv), ptr byref(i128))
fn zeros<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder
        .build_load(llvm_ctx.i128_type(), arg_ptr, "arg")
        .unwrap()
        .into_int_value();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, arg).unwrap();
    builder
        .build_store(ret_bits_ptr, llvm_ctx.i128_type().const_zero())
        .unwrap();
    builder.build_return(None).unwrap();
}

// declare void @"%i64->%i"(ptr sret(i128), ptr byref(i64))
fn i64_to_i<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder
        .build_load(llvm_ctx.i64_type(), arg_ptr, "arg")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_s_extend(arg, llvm_ctx.i128_type(), "sext(arg, 128)")
        .unwrap();
    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @append_64(ptr sret(%bv), ptr byref(%bv), ptr byref(%bv))
fn append<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let upper_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let upper_len_ptr = builder
        .build_struct_gep(types.bitvec, upper_ptr, 0, "&upper.len")
        .unwrap();
    let upper_len = builder
        .build_load(llvm_ctx.i128_type(), upper_len_ptr, "upper.len")
        .unwrap()
        .into_int_value();
    let upper_bits_ptr = builder
        .build_struct_gep(types.bitvec, upper_ptr, 1, "&upper.bits")
        .unwrap();
    let upper_bits = builder
        .build_load(llvm_ctx.i128_type(), upper_bits_ptr, "upper.bits")
        .unwrap()
        .into_int_value();

    let lower_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let lower_len_ptr = builder
        .build_struct_gep(types.bitvec, lower_ptr, 0, "&lower.len")
        .unwrap();
    let lower_len = builder
        .build_load(llvm_ctx.i128_type(), lower_len_ptr, "lower.len")
        .unwrap()
        .into_int_value();
    let lower_bits_ptr = builder
        .build_struct_gep(types.bitvec, lower_ptr, 1, "&lower.bits")
        .unwrap();
    let lower_bits = builder
        .build_load(llvm_ctx.i128_type(), lower_bits_ptr, "lower.bits")
        .unwrap()
        .into_int_value();

    let upper_sh = builder
        .build_left_shift(upper_bits, lower_len, "upper_sh")
        .unwrap();
    let result_bits = builder.build_or(lower_bits, upper_sh, "res").unwrap();

    let result_bv_len = builder
        .build_int_add(lower_len, upper_len, "lower.len + upper.len")
        .unwrap();
    let result_bv_bits = builder
        .build_int_z_extend(result_bits, llvm_ctx.i128_type(), "res_bv_bits")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, result_bv_len).unwrap();
    builder.build_store(ret_bits_ptr, result_bv_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @neg_int(ptr sret(i128), ptr byref(i128))
fn neg_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder
        .build_load(llvm_ctx.i128_type(), arg_ptr, "arg")
        .unwrap()
        .into_int_value();
    let ret = builder.build_int_neg(arg, "-1 * arg").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @add_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn add_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder.build_int_add(lhs, rhs, "sum").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @mult_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn mult_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder.build_int_mul(lhs, rhs, "lhs * rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @shr_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn shr_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_right_shift(lhs, rhs, true, "lhs >> rhs")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @shl_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn shl_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder.build_left_shift(lhs, rhs, "lhs << rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @shr_mach_int(ptr sret(i64), ptr byref(i64), ptr byref(i64))
fn shr_mach_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i64_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i64_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_right_shift(lhs, rhs, true, "lhs >> rhs")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @shl_mach_int(ptr sret(i64), ptr byref(i64), ptr byref(i64))
fn shl_mach_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i64_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i64_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder.build_left_shift(lhs, rhs, "lhs << rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @ediv_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn ediv_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let lhs = builder
        .build_int_truncate(lhs, llvm_ctx.i64_type(), "lhs64")
        .unwrap();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let rhs = builder
        .build_int_truncate(rhs, llvm_ctx.i64_type(), "rhs64")
        .unwrap();

    let ret = builder.build_int_signed_div(lhs, rhs, "lhs / rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @tdiv_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn tdiv_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let lhs = builder
        .build_int_truncate(lhs, llvm_ctx.i64_type(), "lhs64")
        .unwrap();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let rhs = builder
        .build_int_truncate(rhs, llvm_ctx.i64_type(), "rhs64")
        .unwrap();

    let ret = builder.build_int_signed_div(lhs, rhs, "lhs / rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @emod_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn emod_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let lhs = builder
        .build_int_truncate(lhs, llvm_ctx.i64_type(), "lhs64")
        .unwrap();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let rhs = builder
        .build_int_truncate(rhs, llvm_ctx.i64_type(), "rhs64")
        .unwrap();

    let ret = builder.build_int_signed_rem(lhs, rhs, "lhs / rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @tmod_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn tmod_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let lhs = builder
        .build_int_truncate(lhs, llvm_ctx.i64_type(), "lhs64")
        .unwrap();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let rhs = builder
        .build_int_truncate(rhs, llvm_ctx.i64_type(), "rhs64")
        .unwrap();

    let ret = builder.build_int_signed_rem(lhs, rhs, "lhs / rhs").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @max_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn max_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let lhs_lt_rhs = builder
        .build_int_compare(llvm::IntPredicate::SGT, lhs, rhs, "lhs > rhs")
        .unwrap();
    let ret = builder.build_select(lhs_lt_rhs, lhs, rhs, "min").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @min_int(ptr sret(i128), ptr byref(i128), ptr byref(i128))
fn min_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let lhs_lt_rhs = builder
        .build_int_compare(llvm::IntPredicate::SLT, lhs, rhs, "lhs < rhs")
        .unwrap();
    let ret = builder.build_select(lhs_lt_rhs, lhs, rhs, "min").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @gteq(ptr sret(i1), ptr byref(i128), ptr byref(i128))
fn gteq<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_compare(llvm::IntPredicate::SGE, lhs, rhs, "sge")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @add_bits_int(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn add_bits_int<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();

    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, lhs_len, "1 << lhs.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << lhs.len) - 1")
        .unwrap();
    let rhs_bits = builder.build_and(rhs, mask, "rhs[0..lhs.len]").unwrap();
    let ret_bits = builder
        .build_int_add(lhs_bits, rhs_bits, "lhs.bits + rhs[0..lhs.len]")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, lhs_bits).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @add_bits(ptr sret(%bv), ptr byref(%bv), ptr byref(%bv))
fn add_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs_len_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 0, "&rhs.len")
        .unwrap();
    let rhs_len = builder
        .build_load(llvm_ctx.i128_type(), rhs_len_ptr, "rhs.len")
        .unwrap()
        .into_int_value();
    let rhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 1, "&rhs.bits")
        .unwrap();
    let rhs_bits = builder
        .build_load(llvm_ctx.i128_type(), rhs_bits_ptr, "rhs.bits")
        .unwrap()
        .into_int_value();

    let ret_bits = builder
        .build_int_add(lhs_bits, rhs_bits, "lhs.bits + rhs.bits")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, rhs_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sub_bits_int(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn sub_bits_int<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_len_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 0, "&lhs.len")
        .unwrap();
    let lhs_len = builder
        .build_load(llvm_ctx.i128_type(), lhs_len_ptr, "lhs.len")
        .unwrap()
        .into_int_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();

    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, lhs_len, "1 << lhs.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << lhs.len) - 1")
        .unwrap();
    let rhs_bits = builder.build_and(rhs, mask, "rhs[0..lhs.len]").unwrap();
    let ret_bits = builder
        .build_int_sub(lhs_bits, rhs_bits, "lhs.bits - rhs[0..lhs.len]")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, lhs_bits).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sub_bits(ptr sret(%bv), ptr byref(%bv), ptr byref(%bv))
fn sub_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs_len_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 0, "&rhs.len")
        .unwrap();
    let rhs_len = builder
        .build_load(llvm_ctx.i128_type(), rhs_len_ptr, "rhs.len")
        .unwrap()
        .into_int_value();
    let rhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 1, "&rhs.bits")
        .unwrap();
    let rhs_bits = builder
        .build_load(llvm_ctx.i128_type(), rhs_bits_ptr, "rhs.bits")
        .unwrap()
        .into_int_value();

    let ret_bits = builder
        .build_int_sub(lhs_bits, rhs_bits, "lhs.bits - rhs.bits")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, rhs_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @vector_access.56(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn bitvector_access<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    // i128
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let n_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let n = builder
        .build_load(llvm_ctx.i128_type(), n_ptr, "n")
        .unwrap()
        .into_int_value();

    let res_bits = builder
        .build_right_shift(bv_bits, n, false, "bv_bits >> n")
        .unwrap();
    let one = llvm_ctx.i128_type().const_int(1, false);
    let res_bits = builder
        .build_and(res_bits, one, "(bv_bits >> n) & 1")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, one).unwrap();
    builder.build_store(ret_bits_ptr, res_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @and_bits(ptr sret(%bv), ptr byref(%bv), ptr byref(%bv))
fn and_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs_len_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 0, "&rhs.len")
        .unwrap();
    let rhs_len = builder
        .build_load(llvm_ctx.i128_type(), rhs_len_ptr, "rhs.len")
        .unwrap()
        .into_int_value();
    let rhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 1, "&rhs.bits")
        .unwrap();
    let rhs_bits = builder
        .build_load(llvm_ctx.i128_type(), rhs_bits_ptr, "rhs.bits")
        .unwrap()
        .into_int_value();

    let ret_bits = builder
        .build_and(lhs_bits, rhs_bits, "lhs.bits & rhs.bits")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, rhs_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @or_bits(ptr sret(%bv), ptr byref(%bv), ptr byref(%bv))
fn or_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, lhs_ptr, 1, "&lhs.bits")
        .unwrap();
    let lhs_bits = builder
        .build_load(llvm_ctx.i128_type(), lhs_bits_ptr, "lhs.bits")
        .unwrap()
        .into_int_value();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs_len_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 0, "&rhs.len")
        .unwrap();
    let rhs_len = builder
        .build_load(llvm_ctx.i128_type(), rhs_len_ptr, "rhs.len")
        .unwrap()
        .into_int_value();
    let rhs_bits_ptr = builder
        .build_struct_gep(types.bitvec, rhs_ptr, 1, "&rhs.bits")
        .unwrap();
    let rhs_bits = builder
        .build_load(llvm_ctx.i128_type(), rhs_bits_ptr, "rhs.bits")
        .unwrap()
        .into_int_value();

    let ret_bits = builder
        .build_or(lhs_bits, rhs_bits, "lhs.bits | rhs.bits")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, rhs_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @zero_extend.142(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn zero_extend<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let zext_arg_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let zext_arg = builder
        .build_load(llvm_ctx.i128_type(), zext_arg_ptr, "zext_arg")
        .unwrap()
        .into_int_value();

    let one = llvm_ctx.i128_type().const_int(1, false);
    let zero_mask_len = builder
        .build_int_sub(zext_arg, bv_len, "zext_arg - bv.len")
        .unwrap();
    let zero_mask = builder
        .build_left_shift(one, zero_mask_len, "1 << (zext_arg - bv.len)")
        .unwrap();
    let zero_mask = builder
        .build_int_sub(zero_mask, one, "(1 << (zext_arg - bv.len)) - 1")
        .unwrap();
    let zero_mask = builder
        .build_left_shift(
            zero_mask,
            bv_len,
            "((1 << (zext_arg - bv.len)) - 1) << bv.len",
        )
        .unwrap();
    let zero_mask = builder
        .build_not(zero_mask, "!(((1 << (zext_arg - bv.len)) - 1) << bv.len)")
        .unwrap();
    let ret_bits = builder
        .build_and(
            bv_bits,
            zero_mask,
            "bv.bits & !(((1 << (zext_arg - bv.len)) - 1) << bv.len)",
        )
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, zext_arg).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sail_sign_extend(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn sign_extend<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let sext_arg_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let sext_arg = builder
        .build_load(llvm_ctx.i128_type(), sext_arg_ptr, "sext_arg")
        .unwrap()
        .into_int_value();

    // Zero everything above the length
    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, bv_len, "1 << bits.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << bits.len) - 1")
        .unwrap();
    let ret_bits = builder
        .build_and(bv_bits, mask, "bv.bits & ((1 << bits.len) - 1)")
        .unwrap();

    let sign_pos = builder.build_int_sub(bv_len, one, "len - 1").unwrap();
    let sign_bit = builder
        .build_right_shift(bv_bits, sign_pos, false, "bv.bits >> (len - 1)")
        .unwrap();
    let sign_bit = builder
        .build_int_truncate(sign_bit, llvm_ctx.bool_type(), "sign")
        .unwrap();
    let sign_mask = builder
        .build_int_s_extend(sign_bit, llvm_ctx.i128_type(), "sext(sign, 128)")
        .unwrap();
    let sign_mask = builder
        .build_left_shift(sign_mask, bv_len, "sext(sign, 128) << bv.len")
        .unwrap();
    let ret_bits = builder
        .build_or(ret_bits, sign_mask, "ret_bits | sign_mask")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, sext_arg).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @sail_unsigned.291(ptr sret(i128), ptr byref(%bv))
fn sail_unsigned<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let one = llvm_ctx.i128_type().const_int(1, false);
    let mask = builder
        .build_left_shift(one, bv_len, "1 << bits.len")
        .unwrap();
    let mask = builder
        .build_int_sub(mask, one, "(1 << bits.len) - 1")
        .unwrap();
    let ret = builder
        .build_and(bv_bits, mask, "bv.bits & ((1 << bits.len) - 1)")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @eq_int(ptr sret(i1), ptr byref(i128), ptr byref(i128))
fn eq_int<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder
        .build_load(llvm_ctx.i128_type(), lhs_ptr, "lhs")
        .unwrap()
        .into_int_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder
        .build_load(llvm_ctx.i128_type(), rhs_ptr, "rhs")
        .unwrap()
        .into_int_value();
    let ret = builder
        .build_int_compare(llvm::IntPredicate::EQ, lhs, rhs, "eq")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

fn set_slice_internal<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
    bv_ptr: llvm::PointerValue<'a>,
    n_ptr: llvm::PointerValue<'a>,
    update_ptr: llvm::PointerValue<'a>,
    n_is_i64: bool,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let n = if n_is_i64 {
        let n = builder
            .build_load(llvm_ctx.i64_type(), n_ptr, "n")
            .unwrap()
            .into_int_value();
        builder
            .build_int_z_extend(n, llvm_ctx.i128_type(), "n")
            .unwrap()
    } else {
        builder
            .build_load(llvm_ctx.i128_type(), n_ptr, "n")
            .unwrap()
            .into_int_value()
    };

    let update_len_ptr = builder
        .build_struct_gep(types.bitvec, update_ptr, 0, "&update.len")
        .unwrap();
    let update_len = builder
        .build_load(llvm_ctx.i128_type(), update_len_ptr, "update.len")
        .unwrap()
        .into_int_value();
    let update_bits_ptr = builder
        .build_struct_gep(types.bitvec, update_ptr, 1, "&update.bits")
        .unwrap();
    let update_bits = builder
        .build_load(llvm_ctx.i128_type(), update_bits_ptr, "update.bits")
        .unwrap()
        .into_int_value();

    let ret_len = builder
        .build_int_add(n, update_len, "n + update.len")
        .unwrap();
    let ret_len_cmp = builder
        .build_int_compare(
            llvm::IntPredicate::UGT,
            ret_len,
            bv_len,
            "n + update.len > bv.len",
        )
        .unwrap();
    let ret_len = builder
        .build_select(ret_len_cmp, ret_len, bv_len, "ret_len")
        .unwrap();

    let one = llvm_ctx.i128_type().const_int(1, false);
    let zero_mask = builder
        .build_left_shift(one, update_len, "1 << update.len")
        .unwrap();
    let zero_mask = builder
        .build_int_sub(zero_mask, one, "(1 << update.len) - 1")
        .unwrap();
    let zero_mask = builder
        .build_left_shift(zero_mask, n, "((1 << update.len) - 1) << n")
        .unwrap();
    let zero_mask = builder
        .build_not(zero_mask, "!(((1 << update.len) - 1) << n)")
        .unwrap();
    let cleared_bv = builder
        .build_and(bv_bits, zero_mask, "bv.bits & zero_mask")
        .unwrap();
    let update_shifted = builder
        .build_left_shift(update_bits, n, "update.bits << n")
        .unwrap();
    let ret_bits = builder.build_or(cleared_bv, update_shifted, "ret").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, ret_len).unwrap();
    builder.build_store(ret_bits_ptr, ret_bits).unwrap();
    builder.build_return(None).unwrap();
}

fn vector_update_subrange<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let n_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let update_ptr = f.get_nth_param(4).unwrap().into_pointer_value();
    set_slice_internal(f, types, llvm_ctx, bv_ptr, n_ptr, update_ptr, false);
}

fn set_slice<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let bv_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let n_ptr = f.get_nth_param(4).unwrap().into_pointer_value();
    let update_ptr = f.get_nth_param(5).unwrap().into_pointer_value();
    set_slice_internal(f, types, llvm_ctx, bv_ptr, n_ptr, update_ptr, false);
}

// define void @bitvector_update(ptr sret(%bv), ptr byref(%bv), ptr byref(i128), ptr byref(%bv))
// or
// define void @zupdate_fbits(ptr sret(%bv) %0, ptr byref(%bv) %1, ptr byref(i64) %2, ptr byref(%bv) %3) {
fn bitvector_update<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let n_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let update_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    set_slice_internal(f, types, llvm_ctx, bv_ptr, n_ptr, update_ptr, true);
}

// declare void @replicate_bits(ptr sret(%bv), ptr byref(%bv), ptr byref(i128))
fn replicate_bits<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let bv_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let bv_len_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 0, "&bv.len")
        .unwrap();
    let bv_len = builder
        .build_load(llvm_ctx.i128_type(), bv_len_ptr, "bv.len")
        .unwrap()
        .into_int_value();
    let bv_bits_ptr = builder
        .build_struct_gep(types.bitvec, bv_ptr, 1, "&bv.bits")
        .unwrap();
    let bv_bits = builder
        .build_load(llvm_ctx.i128_type(), bv_bits_ptr, "bv.bits")
        .unwrap()
        .into_int_value();

    let times_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let times = builder
        .build_load(llvm_ctx.i128_type(), times_ptr, "times")
        .unwrap()
        .into_int_value();
    let bv_len_times = builder
        .build_int_mul(bv_len, times, "bv.len * times")
        .unwrap();
    let max_len = llvm_ctx.i128_type().const_int(128, false);
    let repl_len = builder
        .build_int_compare(
            llvm::IntPredicate::ULT,
            bv_len_times,
            max_len,
            "bv.len * times < 128",
        )
        .unwrap();
    let repl_len = builder
        .build_select(repl_len, bv_len_times, max_len, "repl_len")
        .unwrap()
        .into_int_value();

    let res_len = builder
        .build_alloca(llvm_ctx.i128_type(), "running_len")
        .unwrap();
    builder
        .build_store(res_len, llvm_ctx.i128_type().const_zero())
        .unwrap();
    let res_bits = builder
        .build_alloca(llvm_ctx.i128_type(), "running_bits")
        .unwrap();
    builder
        .build_store(res_bits, llvm_ctx.i128_type().const_zero())
        .unwrap();
    let count = builder
        .build_alloca(llvm_ctx.i128_type(), "running_count")
        .unwrap();
    builder
        .build_store(count, llvm_ctx.i128_type().const_zero())
        .unwrap();

    let repl_block = llvm_ctx.append_basic_block(f, "repl_block");
    let finish_block = llvm_ctx.append_basic_block(f, "finish_block");
    builder.build_unconditional_branch(repl_block).unwrap();

    builder.position_at_end(repl_block);
    let res_len_val = builder
        .build_load(llvm_ctx.i128_type(), res_len, "running_len_val")
        .unwrap()
        .into_int_value();
    let res_bits_val = builder
        .build_load(llvm_ctx.i128_type(), res_bits, "running_bits_val")
        .unwrap()
        .into_int_value();
    let count_val = builder
        .build_load(llvm_ctx.i128_type(), count, "running_count_val")
        .unwrap()
        .into_int_value();

    let shift = builder
        .build_int_mul(count_val, bv_len, "running_count_val * bv.len")
        .unwrap();
    let res_len_val = builder
        .build_int_add(res_len_val, bv_len, "running_len_val + bv.len")
        .unwrap();
    let replicate_bits_mask = builder
        .build_left_shift(bv_bits, shift, "bv.bits << (running_count_val * bv.len)")
        .unwrap();
    let res_bits_val = builder
        .build_or(
            res_bits_val,
            replicate_bits_mask,
            "running_bits_val | (bv.bits << (running_count_val * bv.len))",
        )
        .unwrap();
    let count_val = builder
        .build_int_add(
            count_val,
            llvm_ctx.i128_type().const_int(1, false),
            "running_count_val + 1",
        )
        .unwrap();

    let stop = builder
        .build_int_compare(llvm::IntPredicate::UGE, res_len_val, repl_len, "stop_cond")
        .unwrap();

    builder.build_store(res_len, res_len_val).unwrap();
    builder.build_store(res_bits, res_bits_val).unwrap();
    builder.build_store(count, count_val).unwrap();
    builder
        .build_conditional_branch(stop, finish_block, repl_block)
        .unwrap();

    builder.position_at_end(finish_block);

    let res_len = builder
        .build_load(llvm_ctx.i128_type(), res_len, "running_len_final")
        .unwrap()
        .into_int_value();
    let res_bits = builder
        .build_load(llvm_ctx.i128_type(), res_bits, "running_bits_final")
        .unwrap()
        .into_int_value();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, res_len).unwrap();
    builder.build_store(ret_bits_ptr, res_bits).unwrap();

    builder.build_return(None).unwrap();
}

// declare void @emulator_read_tag(ptr sret(i1), ptr byref(i64), ptr byref(%bv))
fn emulator_read_tag<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    // TODO: is this ever actually called?
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);
    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder
        .build_store(ret_ptr, llvm_ctx.bool_type().const_zero())
        .unwrap();
    builder.build_return(None).unwrap();
}

// declare void @monomorphize(ptr sret(%bv), ptr byref(%bv))
fn monomorphize<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);
    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder.build_load(types.bitvec, arg_ptr, "arg").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, arg).unwrap();
    builder.build_return(None).unwrap();
}

fn vector_access<'a>(
    f: llvm::FunctionValue<'a>,
    param_pointee_tys: &[llvm::BasicMetadataTypeEnum],
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    if let llvm::BasicMetadataTypeEnum::StructType(ty) = param_pointee_tys[1] {
        if ty == types.bitvec {
            bitvector_access(f, types, llvm_ctx);
            return;
        } else {
            panic!("vector_access for {ty}");
        }
    }

    let vec_ty = param_pointee_tys[1].into_array_type();

    // build a monomorphized vector_access for whatever types are in the given param
    // pointee types list
    // declare void @fn_name(ptr sret(T), ptr byref(ptr), ptr byref(i128))
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let ptr_to_global_vec_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let global_vec_ptr = builder
        .build_load(
            llvm_ctx.ptr_type(llvm::AddressSpace::default()),
            ptr_to_global_vec_ptr,
            "global vec ptr",
        )
        .unwrap()
        .into_pointer_value();

    let elem_idx_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let elem_idx = builder
        .build_load(llvm_ctx.i128_type(), elem_idx_ptr, "elem_idx")
        .unwrap()
        .into_int_value();
    let elem_idx = builder
        .build_int_truncate(elem_idx, llvm_ctx.i32_type(), "elem_idx32")
        .unwrap();

    let elem_ptr = unsafe {
        builder
            .build_gep(
                vec_ty,
                global_vec_ptr,
                &[llvm_ctx.i32_type().const_zero(), elem_idx],
                "elem_ptr",
            )
            .unwrap()
    };
    let new_elem = builder
        .build_load(vec_ty.get_element_type(), elem_ptr, "new_elem")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, new_elem).unwrap();
    builder.build_return(None).unwrap();
}

fn vector_update<'a>(
    f: llvm::FunctionValue<'a>,
    param_pointee_tys: &[llvm::BasicMetadataTypeEnum],
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    if let llvm::BasicMetadataTypeEnum::StructType(ty) = param_pointee_tys[1] {
        if ty == types.bitvec {
            bitvector_update(f, types, llvm_ctx);
            return;
        } else {
            panic!("vector_update for {ty}");
        }
    }

    let vec_ty = param_pointee_tys[1].into_array_type();

    // build a monomorphized vector_update for whatever types are in the given param
    // pointee types list
    // declare void @fn_name(ptr sret(ptr), ptr byref(ptr), ptr byref(i128), ptr byref(T))
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let ptr_to_global_vec_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let global_vec_ptr = builder
        .build_load(
            llvm_ctx.ptr_type(llvm::AddressSpace::default()),
            ptr_to_global_vec_ptr,
            "global vec ptr",
        )
        .unwrap()
        .into_pointer_value();

    let elem_idx_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let elem_idx = builder
        .build_load(llvm_ctx.i128_type(), elem_idx_ptr, "elem_idx")
        .unwrap()
        .into_int_value();
    let elem_idx = builder
        .build_int_truncate(elem_idx, llvm_ctx.i32_type(), "elem_idx32")
        .unwrap();

    let new_elem_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let new_elem = builder
        .build_load(vec_ty.get_element_type(), new_elem_ptr, "update_elem")
        .unwrap();

    let elem_ptr = unsafe {
        builder
            .build_gep(
                vec_ty,
                global_vec_ptr,
                &[llvm_ctx.i32_type().const_zero(), elem_idx],
                "elem_ptr",
            )
            .unwrap()
    };
    builder.build_store(elem_ptr, new_elem).unwrap();

    // Return is unused, don't modify it
    builder.build_return(None).unwrap();
}

fn list_cons_or_internal_pick<'a>(
    f: llvm::FunctionValue<'a>,
    param_pointee_tys: &[llvm::BasicMetadataTypeEnum],
    llvm_ctx: &'a llvm::Context,
) {
    let ret_ty = param_pointee_tys[0].into_int_type();
    let arg_ty = param_pointee_tys[1].into_int_type();
    assert_eq!(ret_ty, arg_ty);

    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let arg_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let arg = builder.build_load(arg_ty, arg_ptr, "arg").unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, arg).unwrap();
    builder.build_return(None).unwrap();
}

// declare void @eq_string(ptr sret(i1), ptr byref([0 x i8]), ptr byref([0 x i8]))
fn eq_string<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let entry_block = llvm_ctx.append_basic_block(f, "entry");
    let comparison_block = llvm_ctx.append_basic_block(f, "comparison");
    let continue_block = llvm_ctx.append_basic_block(f, "continue");
    let equal_block = llvm_ctx.append_basic_block(f, "equal");
    let not_equal_block = llvm_ctx.append_basic_block(f, "not_equal");
    let finish_block = llvm_ctx.append_basic_block(f, "finish");

    builder.position_at_end(entry_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();

    let cmp_idx = builder
        .build_alloca(llvm_ctx.i32_type(), "cmp_idx")
        .unwrap();
    builder
        .build_store(cmp_idx, llvm_ctx.i32_type().const_zero())
        .unwrap();
    builder
        .build_unconditional_branch(comparison_block)
        .unwrap();

    // Comparison block
    builder.position_at_end(comparison_block);
    let cmp_idx_val = builder
        .build_load(llvm_ctx.i32_type(), cmp_idx, "cmp_idx_val")
        .unwrap()
        .into_int_value();
    let lhs_char = unsafe {
        builder.build_gep(
            llvm_ctx.i8_type().array_type(0),
            lhs_ptr,
            &[llvm_ctx.i32_type().const_zero(), cmp_idx_val],
            "lhs_char",
        )
    }
    .unwrap();
    let rhs_char = unsafe {
        builder.build_gep(
            llvm_ctx.i8_type().array_type(0),
            rhs_ptr,
            &[llvm_ctx.i32_type().const_zero(), cmp_idx_val],
            "rhs_char",
        )
    }
    .unwrap();

    let lhs_char_val = builder
        .build_load(llvm_ctx.i8_type(), lhs_char, "lhs_char_val")
        .unwrap()
        .into_int_value();
    let rhs_char_val = builder
        .build_load(llvm_ctx.i8_type(), rhs_char, "rhs_char_val")
        .unwrap()
        .into_int_value();

    let cmp_idx_val = builder
        .build_int_add(
            cmp_idx_val,
            llvm_ctx.i32_type().const_int(1, false),
            "next_cmp_idx_val",
        )
        .unwrap();
    builder.build_store(cmp_idx, cmp_idx_val).unwrap();

    let eq_cmp = builder
        .build_int_compare(
            llvm::IntPredicate::EQ,
            lhs_char_val,
            rhs_char_val,
            "lhs_char_val == rhs_char_val",
        )
        .unwrap();

    builder
        .build_conditional_branch(eq_cmp, continue_block, not_equal_block)
        .unwrap();

    // continue_block
    builder.position_at_end(continue_block);
    let chars_are_null = builder
        .build_int_compare(
            llvm::IntPredicate::EQ,
            rhs_char_val,
            llvm_ctx.i8_type().const_zero(),
            "chars_are_null",
        )
        .unwrap();
    builder
        .build_conditional_branch(chars_are_null, equal_block, comparison_block)
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();

    // equal_block
    builder.position_at_end(equal_block);
    builder
        .build_store(ret_ptr, llvm_ctx.bool_type().const_all_ones())
        .unwrap();
    builder.build_unconditional_branch(finish_block).unwrap();

    // not_equal_block
    builder.position_at_end(not_equal_block);
    builder
        .build_store(ret_ptr, llvm_ctx.bool_type().const_zero())
        .unwrap();
    builder.build_unconditional_branch(finish_block).unwrap();

    // finish
    builder.position_at_end(finish_block);
    builder.build_return(None).unwrap();
}

fn eq_anything<'a>(
    f: llvm::FunctionValue<'a>,
    param_pointee_tys: &[llvm::BasicMetadataTypeEnum],
    llvm_ctx: &'a llvm::Context,
) {
    let lhs_ty = llvm::BasicTypeEnum::try_from(param_pointee_tys[1]).unwrap();
    let rhs_ty = llvm::BasicTypeEnum::try_from(param_pointee_tys[2]).unwrap();

    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let lhs_ptr = f.get_nth_param(1).unwrap().into_pointer_value();
    let lhs = builder.build_load(lhs_ty, lhs_ptr, "lhs").unwrap();

    let rhs_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let rhs = builder.build_load(rhs_ty, rhs_ptr, "rhs").unwrap();

    let ret = llvm_deep_eq(lhs, rhs, &builder, llvm_ctx);

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder.build_store(ret_ptr, ret).unwrap();
    builder.build_return(None).unwrap();
}

fn do_nothing<'a>(f: llvm::FunctionValue<'a>, llvm_ctx: &'a llvm::Context) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);
    builder.build_return(None).unwrap();
}

fn read_mem<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
    zinq_mem_map: llvm::GlobalValue<'a>,
    zinq_read_mem: llvm::FunctionValue<'a>,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let addr_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let addr_bits_ptr = builder
        .build_struct_gep(types.bitvec, addr_ptr, 1, "&addr.bits")
        .unwrap();
    let addr_bits = builder
        .build_load(llvm_ctx.i128_type(), addr_bits_ptr, "addr.bits")
        .unwrap()
        .into_int_value();
    let addr_bits = builder
        .build_int_truncate(addr_bits, llvm_ctx.i64_type(), "addr64")
        .unwrap();

    let bytes_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let bytes = builder
        .build_load(llvm_ctx.i128_type(), bytes_ptr, "bytes")
        .unwrap()
        .into_int_value();
    let bytes = builder
        .build_int_truncate(bytes, llvm_ctx.i64_type(), "bytes64")
        .unwrap();

    let read_mem_res = builder
        .build_call(
            zinq_read_mem,
            &[
                zinq_mem_map.as_pointer_value().into(),
                addr_bits.into(),
                bytes.into(),
            ],
            "read_mem_res",
        )
        .unwrap()
        .try_as_basic_value()
        .left()
        .unwrap()
        .into_int_value();
    let read_mem_len = builder
        .build_int_mul(
            bytes,
            llvm_ctx.i64_type().const_int(8, false),
            "read_mem_len",
        )
        .unwrap();

    let read_mem_res = builder
        .build_int_z_extend(read_mem_res, llvm_ctx.i128_type(), "read_mem_res128")
        .unwrap();
    let read_mem_len = builder
        .build_int_z_extend(read_mem_len, llvm_ctx.i128_type(), "read_mem_len128")
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    let ret_len_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 0, "ret_len_ptr")
        .unwrap();
    let ret_bits_ptr = builder
        .build_struct_gep(types.bitvec, ret_ptr, 1, "ret_bits_ptr")
        .unwrap();
    builder.build_store(ret_len_ptr, read_mem_len).unwrap();
    builder.build_store(ret_bits_ptr, read_mem_res).unwrap();
    builder.build_return(None).unwrap();
}

fn write_mem<'a>(
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
    zinq_mem_map: llvm::GlobalValue<'a>,
    zinq_write_mem: llvm::FunctionValue<'a>,
) {
    let builder = llvm_ctx.create_builder();
    let basic_block = llvm_ctx.append_basic_block(f, "entry");
    builder.position_at_end(basic_block);

    let addr_ptr = f.get_nth_param(2).unwrap().into_pointer_value();
    let addr_bits_ptr = builder
        .build_struct_gep(types.bitvec, addr_ptr, 1, "&addr.bits")
        .unwrap();
    let addr_bits = builder
        .build_load(llvm_ctx.i128_type(), addr_bits_ptr, "addr.bits")
        .unwrap()
        .into_int_value();
    let addr_bits = builder
        .build_int_truncate(addr_bits, llvm_ctx.i64_type(), "addr64")
        .unwrap();

    let data_ptr = f.get_nth_param(3).unwrap().into_pointer_value();
    let data_len_ptr = builder
        .build_struct_gep(types.bitvec, data_ptr, 1, "&data.len")
        .unwrap();
    let data_len = builder
        .build_load(llvm_ctx.i128_type(), data_len_ptr, "data.len")
        .unwrap()
        .into_int_value();
    let data_len = builder
        .build_int_truncate(data_len, llvm_ctx.i64_type(), "data_len64")
        .unwrap();
    let data_bits_ptr = builder
        .build_struct_gep(types.bitvec, data_ptr, 1, "&data.bits")
        .unwrap();
    let data_bits = builder
        .build_load(llvm_ctx.i128_type(), data_bits_ptr, "data.bits")
        .unwrap()
        .into_int_value();
    let data_bits = builder
        .build_int_truncate(data_bits, llvm_ctx.i64_type(), "data_bits64")
        .unwrap();

    builder
        .build_call(
            zinq_write_mem,
            &[
                zinq_mem_map.as_pointer_value().into(),
                addr_bits.into(),
                data_bits.into(),
                data_len.into(),
            ],
            "write_mem_res",
        )
        .unwrap();

    let ret_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
    builder
        .build_store(ret_ptr, llvm_ctx.bool_type().const_all_ones())
        .unwrap();
    builder.build_return(None).unwrap();
}

pub fn fill_if_mem_access<'a>(
    name: &str,
    f: llvm::FunctionValue<'a>,
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
    zinq_mem_map: llvm::GlobalValue<'a>,
    zinq_read_mem: llvm::FunctionValue<'a>,
    zinq_write_mem: llvm::FunctionValue<'a>,
) {
    match name {
        "read_mem" | "read_mem_ifetch" | "read_mem_exclusive" => {
            read_mem(f, types, &llvm_ctx, zinq_mem_map, zinq_read_mem)
        }
        "write_mem" | "write_mem_exclusive" => {
            write_mem(f, types, &llvm_ctx, zinq_mem_map, zinq_write_mem)
        }
        _ => {}
    }
}

pub fn fill_if_supported<'a>(
    name: &str,
    f: llvm::FunctionValue<'a>,
    param_pointee_tys: &[llvm::BasicMetadataTypeEnum],
    types: &J2LDeclaredTypes<'a>,
    llvm_ctx: &'a llvm::Context,
) {
    match name {
        "vector_access" => vector_access(f, param_pointee_tys, types, llvm_ctx),
        "vector_update_subrange" => vector_update_subrange(f, types, llvm_ctx),
        "vector_subrange" => vector_subrange(f, types, llvm_ctx),
        "vector_update" => vector_update(f, param_pointee_tys, types, llvm_ctx),
        "cons" | "internal_pick" => list_cons_or_internal_pick(f, param_pointee_tys, llvm_ctx),
        "pow2" => pow2(f, llvm_ctx),
        "not" => not(f, llvm_ctx),
        "undefined_bitvector" => undefined_bitvector(f, types, llvm_ctx),
        "undefined_bool" => undefined_bool(f, llvm_ctx),
        "undefined_int" => undefined_int(f, llvm_ctx),
        "not_bits" => not_bits(f, types, llvm_ctx),
        "lteq" => lteq(f, llvm_ctx),
        "shiftl" => shiftl(f, types, llvm_ctx),
        "eq_bits" | "eq_bit" => eq_bits(f, types, llvm_ctx),
        "neq_bits" => neq_bits(f, types, llvm_ctx),
        "lt" => lt(f, llvm_ctx),
        "gt" => gt(f, llvm_ctx),
        "read_register_from_vector" => read_register_from_vector(f, types, llvm_ctx),
        "sail_signed" => sail_signed(f, types, llvm_ctx),
        "slice" => slice(f, types, llvm_ctx),
        "get_slice_int" => get_slice_int(f, types, llvm_ctx),
        "sail_truncate" => sail_truncate(f, types, llvm_ctx),
        "sub_int" => sub_int(f, llvm_ctx),
        "%i->%i64" => i_to_i64(f, llvm_ctx),
        "write_register_from_vector" => write_register_from_vector(f, types, llvm_ctx),
        "length" => length(f, types, llvm_ctx),
        "shiftr" => shiftr(f, types, llvm_ctx),
        "arith_shiftr" => arith_shiftr(f, types, llvm_ctx),
        "zeros" => zeros(f, types, llvm_ctx),
        "%i64->%i" => i64_to_i(f, llvm_ctx),
        "append" | "append_64" => append(f, types, llvm_ctx),
        "neg_int" => neg_int(f, llvm_ctx),
        "add_int" => add_int(f, llvm_ctx),
        "mult_int" => mult_int(f, llvm_ctx),
        "ediv_int" => ediv_int(f, llvm_ctx),
        "tdiv_int" => tdiv_int(f, llvm_ctx),
        "emod_int" => emod_int(f, llvm_ctx),
        "tmod_int" => tmod_int(f, llvm_ctx),
        "shl_int" => shl_int(f, llvm_ctx),
        "shr_int" => shr_int(f, llvm_ctx),
        "shl_mach_int" => shl_mach_int(f, llvm_ctx),
        "shr_mach_int" => shr_mach_int(f, llvm_ctx),
        "gteq" => gteq(f, llvm_ctx),
        "add_bits" => add_bits(f, types, llvm_ctx),
        "add_bits_int" => add_bits_int(f, types, llvm_ctx),
        "sub_bits" => sub_bits(f, types, llvm_ctx),
        "sub_bits_int" => sub_bits_int(f, types, llvm_ctx),
        "and_bits" => and_bits(f, types, llvm_ctx),
        "or_bits" => or_bits(f, types, llvm_ctx),
        "sign_extend" => sign_extend(f, types, llvm_ctx),
        "zero_extend" => zero_extend(f, types, llvm_ctx),
        "sail_unsigned" => sail_unsigned(f, types, llvm_ctx),
        "eq_string" => eq_string(f, llvm_ctx),
        "eq_bool" => eq_bool(f, llvm_ctx),
        "eq_int" => eq_int(f, llvm_ctx),
        "max_int" => max_int(f, llvm_ctx),
        "min_int" => min_int(f, llvm_ctx),
        "eq_anything" => eq_anything(f, param_pointee_tys, llvm_ctx),
        "zupdate_fbits" => bitvector_update(f, types, llvm_ctx),
        "set_slice" => set_slice(f, types, llvm_ctx),
        "emulator_read_tag" => emulator_read_tag(f, llvm_ctx),
        "monomorphize" => monomorphize(f, types, llvm_ctx),
        "replicate_bits" => replicate_bits(f, types, llvm_ctx),
        "prerr_bits" | "prerr_int" | "emulator_write_tag" | "sail_putchar" | "branch_announce"
        | "zsail_assert" | "concat_str" | "print_endline" => do_nothing(f, llvm_ctx),
        _ => {}
    }
}
