/*
   FALCON - The Falcon Programming Language.
   FILE: flc_pcodes.h

   The list of byte cods
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun giu 21 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_PCODES_H
#define flc_PCODES_H

#define FALCON_PCODE_VERSION  2
#define FALCON_PCODE_MINOR  2

/** \page opcode_format Virtual machine opcode format

Opcodes are formed by exactly fuor bytes. The first byte is the Op code ID,
and uniquely identifies the instruction that must be performed. The remaining
three bytes are the operator type an can be:

   - \b 00H - not used
   - \b 01H - an immediate integer (32 bit)
   - \b 02H - a string ID (32 bit)
   - \b 03H - an immediate double value (64 bit)
   - \b 04H - an immediate NIL value
   - \b 05H - an immediate 64 bit integer
   - \b 06H - Variable ID in the global table
   - \b 07H - Variable ID in the local table
   - \b 08H - Variable ID in the parameter table
   - \b 09H - Reserved for external symbol load
   - \b 0AH - a string ID representing a late binding
   - \b 0BH - True
   - \b 0CH - False
   - \b 0EH - 32 bit not to be decoded
   - \b 0FH - 64 bit not to be decoded
   - \b 10H - register A - Accumulator
   - \b 11H - register B - Extra
   - \b 12H - register S1 - self
   - \b 13H - register S2 - sender (to be removed)
   - \b 14H - register L1 - first latch (accessed item)
   - \b 15H - register L2 - second latch (accessor item)

The paramters immediately follow the opcode and are stored in little endian order,
if being integer, or directly in float format if being double. If the relative
parameter indicator is 0E or 0F the PC is moved accordingly, but the paramters
are left undecoded; this is useful for instructions that take a fixed parameter
type to avoid variable encoding/decoding. Otherwise, the parameter are decoded
and stored in the VM volatile registers OP1, OP2 and OP3 depending on their
position.

In example, a CALL 3, $func, where $func is stored in the global variable table
at ID 15H, look like this in hex:

<pre>
   3A 0E 05 00   03 00 00 00   15 00 00 00
</pre>

The first 3A is the CALL opcode, then a 0E ordering the VM not to read the first
parameter (which will be handled directly by the CALL handler), but just to
skip 32 bits, then the 05 informing the VM that OP2 is a variable in the global
table, and the 00 indicating that there's no third parameter. This is followed
by "3" and "15H", both stored in big endian order. The VM won't fill 0P1 register,
while it will fill OP2 with an integer item containing 15H.

*/

#define P_PARAM_NOTUSED 0x00
#define P_PARAM_INT32   0x01
#define P_PARAM_STRID   0x02
#define P_PARAM_NUM     0x03
#define P_PARAM_NIL     0x04
#define P_PARAM_INT64   0x05
#define P_PARAM_GLOBID  0x06
#define P_PARAM_LOCID   0x07
#define P_PARAM_PARID   0x08
#define P_PARAM_LBIND   0x0A
#define P_PARAM_TRUE    0x0B
#define P_PARAM_FALSE   0x0C
#define P_PARAM_NTD64   0x0F
#define P_PARAM_NTD32   0x0E
#define P_PARAM_NTD64   0x0F
#define P_PARAM_REGA    0x10
#define P_PARAM_REGB    0x11
#define P_PARAM_REGS1   0x12
#define P_PARAM_REGS2   0x13
#define P_PARAM_REGL1   0x14
#define P_PARAM_REGL2   0x15

// Range 1: parameterless ops
/** END: terminate. */
#define P_END         0x00
/** NOP: No operation. */
#define P_NOP         0x01
/** PSHN: Push a NIL on top of stack. */
#define P_PSHN        0x02
/** RET: Return from a function, nil -> A. */
#define P_RET         0x03
/** RET: Return from a function, assuming A is the return value. */
#define P_RETA        0x04
/** LNIL: Load nil into the parameter, nil -> OP1. */
#define P_LNIL        0x05

// Range 2: one parameter ops

/** RET: Pop last N TRY position from the TRY stack. */
#define P_PTRY        0x06

/** RETV: Return from a function and sets A to the operand, OP1 -> A. */
#define P_RETV        0x07
/** BOOL: Check is the parameter is true or false, and sets A to 0 or 1. */
#define P_BOOL        0x08
/** JMP \<int32\>: Jumps at given position, OP1 -> PC. */
#define P_JMP         0x09
/** GENA: Generates an array given the OP1 previous elements in the stack. */
#define P_GENA        0x0A
/** GEND: Generates a dictionray given the OP1*2 previous elements in the stack. */
#define P_GEND        0x0B
/** PUSH: pushes OP1 in the stack. */
#define P_PUSH        0x0C
/** PSHR: pushes a reference to OP1 in the stack. */
#define P_PSHR        0x0D
/** POP: pops OP1 from the stack. */
#define P_POP         0x0E
/** INC: Inc prefix
   INC OP1: OP1 + 1 -> OP1, OP1-> A
*/
#define P_INC         0x0F
/** DEC: Dec prefix
   DEC OP1: OP1 - 1 -> OP1, OP1-> A
*/
#define P_DEC         0x10
/** NEG: Negates OP1, - OP1 -> OP1. */
#define P_NEG         0x11
/** NOT: sets OP1 to 0 if it's true, and to 1 if it's false. */
#define P_NOT         0x12
/** TRAL \<int32\>: Traverse last. */
#define P_TRAL        0x13
/** IPOP: Pops last OP1 elements from the stack. */
#define P_IPOP        0x14
/** XPOP: exchange OP1 and the topmost stack element. */
#define P_XPOP        0x15

/** GEOR: Generates an open range object [OP1,*[ -> A */
#define P_GEOR        0x16

/** TRY: push OP1 in the try stack. */
#define P_TRY         0x17
/** JTRY: jump out of a try, going at OP1 and popping the try stack. */
#define P_JTRY        0x18
/** RIS: Raise an exception whose value is OP1. */
#define P_RIS         0x19

/** BNOT: Binary NOT ( ~OP1 ) -> A */
#define P_BNOT        0x1A

/** NOTS: Binary self NOT ( ~OP1 ) -> OP1 */
#define P_NOTS        0x1B

/** PEEK: peeks the stack top and copies it in the operand. STACK[top] -> OP1 */
#define P_PEEK        0x1C

// Range3: Double parameter ops

/** FORK: Creates a coroutine jumping at OP2 position.
   OP1 is the amount of stack elements to be passed to the new thread.
   Both ops are fixed
*/
#define P_FORK        0x1D

/** LD: Loads OP1 in OP2  OP1 -> OP2 */
#define P_LD          0x1E
/** LDRF: Loads a reference to OP1 in OP2, &OP1 -> OP2.
   IF OP2 is immediate, then break the reference pointed by OP1
   \todo: add opcode to break reference.
*/
#define P_LDRF        0x1F
/** ADD: OP1 + OP2 -> A */
#define P_ADD         0x20
/** SUB: OP1 - OP2 -> A */
#define P_SUB         0x21
/** MUL: OP1 * OP2 -> A */
#define P_MUL         0x22
/** DIV: OP1 / OP2 -> A */
#define P_DIV         0x23
/** MOD: OP1 % OP2 -> A */
#define P_MOD         0x24
/** POW: OP1^OP2 -> A */
#define P_POW         0x25
/** ADDS: add to self, OP1 + OP2 -> OP1 */
#define P_ADDS        0x26
/** SUBS: subtract from self, OP1 - OP2 -> OP1 */
#define P_SUBS        0x27
/** MULS: multiply to self, OP1 * OP2 -> OP1 */
#define P_MULS        0x28
/** DIVS: divide from self, OP1 / OP2 -> OP1 */
#define P_DIVS        0x29
/** MODS: self module, OP1 % OP2 -> OP1 */
#define P_MODS        0x2A

/** BAND: Binary and ( OP1 & OP2 ) -> A */
#define P_BAND        0x2B
/** BOR: Binary OR ( OP1 | OP2 ) -> A */
#define P_BOR         0x2C
/** BXOR: Binary XOR ( OP1 ^ OP2 ) -> A */
#define P_BXOR        0x2D

/** ANDS: self binary and ( OP1 & OP2 ) -> OP1 */
#define P_ANDS        0x2E
/** ORS: self binary OR ( OP1 | OP2 ) -> OP1 */
#define P_ORS         0x2F
/** XORS: self binary XOR ( OP1 ^ OP2 ) -> OP1 */
#define P_XORS        0x30

/** GENR: Generates an range object [OP1, OP2] -> A */
#define P_GENR        0x31

/** EQ: Compares OP1 and OP2, and if they are equal 1 -> A, else 0 -> A */
#define P_EQ          0x32
/** NEQ: Compares OP1 and OP2, and if they are equal 0 -> A, else 1 -> A */
#define P_NEQ         0x33
/** GT: Compares OP1 and OP2, and if OP1 \> OP2 1 -> A, else 0 -> A */
#define P_GT          0x34
/** GE: Compares OP1 and OP2, and if OP1 \>= OP2 1 -> A, else 0 -> A */
#define P_GE          0x35
/** LT: Compares OP1 and OP2, and if OP1 \<  OP2 1 -> A, else 0 -> A */
#define P_LT          0x36
/** LE: Compares OP1 and OP2, and if OP1 \<= OP2 1 -> A, else 0 -> A */
#define P_LE          0x37

/** IFT \<int32\>, $sym
   If OP2 is true then jumps to OP1 (OP1 -> PC). */
#define P_IFT         0x38
/** IFF \<int32\>, $sym
   If OP2 is false then jumps to OP1 (OP1 -> PC). */
#define P_IFF         0x39
/** CALL \<int32\>, $sym.
   calls subroutine at OP2; at subroutine returns, removes OP1 params from the stack.
*/
#define P_CALL        0x3A
/** INST $sym, \<int32\>.
   As call, but does not changes self and sender registers. */
#define P_INST        0x3B
/** ONCE: Execute a code portion only once.
   If the function OP2 has not been executed, then proceeds, else jumps at OP1.
*/
#define P_ONCE        0x3C

/** LDV: Load from vector OP1 the item at OP2 in A, OP1[ OP2 ] -> A  */
#define P_LDV         0x3D
/** LDV: Load from object OP1 the property OP2 in A, OP1.OP2 -> A  */
#define P_LDP         0x3E

/** TRAN \<int32\>, \<int32\>, \<int32\>
   Traverse next.
   First parameter is the loop point (begin of the loop), second is the loop end
   (loop cleanup sequence). If third parameter is NTD32 == 0, the TRAV iterator is advanced,
   if it's 1 the current element is removed and iterator is updated.
   If this was the last element of the collection, IP is set to param2, else it's set to param1.
*/
#define P_TRAN        0x3F

/** UNPK: unpack vector OP1 into vector OP2. */
#define P_UNPK        0x40

/** SWCH: switch.
   \todo explain
*/
#define P_SWCH        0x41

/** HAS: sets A to 1 if OP1 object has attribute OP2, otherwise sets A to 0. */
#define P_HAS         0x42
/** HAS: sets A to 0 if OP1 object has attribute OP2, otherwise sets A to 0. */
#define P_HASN        0x43
/** GIVE: gives attribute set in OP1 to object OP2. */
#define P_GIVE        0x44
/** GIVN: removes attribute set in OP1 from object OP2. */
#define P_GIVN        0x45
/** IN: sets A to 1 if OP1 is in set OP2, else sets A to 0 */
#define P_IN          0x46
/** NOIN: sets A to 0 if OP1 is in set OP2, else sets A to 1 */
#define P_NOIN        0x47
/** PROV: sets A to 1 if OP1 has property OP2, else sets A to 0 */
#define P_PROV        0x48

/** STVS: Store the topmost stack element into property OP2 of OP1: STACK -> OP1.OP2 */
#define P_STVS        0x49
/** STPS: Store the topmost stack element into position OP2 of OP1: STACK -> OP1[OP2] */
#define P_STPS        0x4A

/** AND: Logical AND if (op1 is true and op2 is true ) true->A */
#define P_AND         0x4B
/** AND: Logical OR if (op1 is true or op2 is true ) true->A  */
#define P_OR          0x4C

/** PASS: calls the given callable object without changing the stack frame. */
#define P_PASS        0x4D

/** PSIN: as PASS, but changes the stack frame (just passes the parameters as they are). */
#define P_PSIN        0x4E

// Range 4: ternary opcodes

/** STV: Store OP3 into vector OP1 at position OP2,  OP3 -> OP1[OP2] */
#define P_STV         0x4F
/** STP: Store OP3 into Property OP2 of OP1: OP3 -> OP1.OP2 */
#define P_STP         0x50
/** LDVT: Load from vector OP1 the item at OP2 in OP3, OP1[ OP2 ] -> OP3  */
#define P_LDVT        0x51
/** LDPT: Load from object OP1 the property OP2 in OP3, OP1.OP2 -> OP3  */
#define P_LDPT        0x52
/** STVR: store reference to OP3 into vector OP1 at pos OP2:  &OP3 -> OP1[OP2] */
#define P_STVR        0x53
/** STPR: store reference to OP3 into property OP2 of OP1:  &OP3 -> OP1.OP2 */
#define P_STPR        0x54

/** TRAV is a ternary opcode;
   TRAV DEST, SOURCE, jmp_position.
   It push 3 items in the stack ( a number, the destination and the source )

*/
#define P_TRAV        0x55

/** INC postfix
   INCP V : V->A, V := V + 1 -> B
*/

#define P_INCP        0x56

/** DEC Postfix
   DECP V :  V->A, V := V - 1 -> B
*/
#define P_DECP        0x57

/** Shift left op1 of op2 positions and place the result in A*/
#define P_SHL         0x58
/** Shift right op1 of op2 positions and place the result in A*/
#define P_SHR         0x59
/** Shift left op1 of op2 positions and place the result in OP1*/
#define P_SHLS        0x5A
/** Shift right op1 of op2 positions and place the result in OP1*/
#define P_SHRS        0x5B

/** Load vector item reference. $OP1[OP2] --> A */
#define P_LDVR        0x5C

/** Load property reference. $OP1.OP2 --> A */
#define P_LDPR        0x5D

/** POWS: self power, OP1 ** OP2 -> OP1 */
#define P_POWS        0x5E

/** Load from byte or character from sting  */
#define P_LSB         0x5F

/** UNPK: unpack vector OP2 into OP1 reference variables int the stack. */
#define P_UNPS        0x60

/** SELECT: Select a branch depending on a variable type. Similar to switch. */
#define P_SELE        0x61

/** INDI: Indirect symbol reference; get the value of the string OP1 */
#define P_INDI        0x62

/** STEX: String expansion; expand string in OP1. */
#define P_STEX        0x63

/** TRAC: Traverse change. Change current (value) item in traversal loops to shallow copy of OP1 */
#define P_TRAC        0x64

/** WRT: Write on standard output (TODO: also other vm streams using parameters) */
#define P_WRT         0x65

/** STO:
   STO OP1, OP2 -> OP1 := OP2.
   Works as LD, but it overwrites target value even if it's a reference.
*/
#define P_STO         0x66

/** FORB
   Forward binding
   FORB OP1, OP2 -> A := lbind( OP1, OP2 ).
*/
#define P_FORB         0x67

#define FLC_PCODE_COUNT 0x68

#endif

/* end of flc_pcodes.h */
