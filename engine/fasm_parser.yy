/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.ypp

   Bison grammar definition for falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven mag 21 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


%{
#include <falcon/setup.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>

#include <fasm/comp.h>
#include <fasm/clexer.h>
#include <fasm/pseudo.h>
#include <falcon/string.h>
#include <falcon/pcodes.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::AsmCompiler *>(fasm_param) )
#define  LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM fasm_param
#define YYLEX_PARAM fasm_param
#define YYSTYPE Falcon::Pseudo *

int fasm_parse( void *param );
void fasm_error( const char *s );

inline int yylex (void *lvalp, void *fasm_param)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__

%}

%pure_parser
%defines
%name-prefix="fasm_"

%token EOL
%token NAME
%token COMMA
%token COLON

/* No conflicts in assembly */
%expect 0

%token DENTRY
%token DGLOBAL
%token DVAR
%token DCONST
%token DATTRIB
%token DLOCAL
%token DPARAM
%token DFUNCDEF
%token DENDFUNC
%token DFUNC
%token DMETHOD
%token DPROP
%token DPROPREF
%token DCLASS
%token DCLASSDEF
%token DCTOR
%token DENDCLASS
%token DFROM
%token DEXTERN
%token DMODULE
%token DLOAD
%token DIMPORT
%token DALIAS
%token DSWITCH
%token DSELECT
%token DCASE
%token DENDSWITCH
%token DLINE
%token DSTRING
%token DISTRING
%token DCSTRING
%token DHAS
%token DHASNT
%token DINHERIT
%token DINSTANCE


%token SYMBOL
%token EXPORT
%token LABEL
%token INTEGER
%token REG_A
%token REG_B
%token REG_S1
%token REG_S2
%token REG_L1
%token REG_L2
%token NUMERIC
%token STRING
%token STRING_ID
%token TRUE_TOKEN
%token FALSE_TOKEN

%token I_LD
%token I_LNIL
%token NIL
%token I_ADD
%token I_SUB
%token I_MUL
%token I_DIV
%token I_MOD
%token I_POW
%token I_ADDS
%token I_SUBS
%token I_MULS
%token I_DIVS
%token I_POWS
%token I_INC
%token I_DEC
%token I_INCP
%token I_DECP
%token I_NEG
%token I_NOT
%token I_RET
%token I_RETV
%token I_RETA
%token I_FORK
%token I_PUSH
%token I_PSHR
%token I_PSHN
%token I_POP
%token I_LDV
%token I_LDVT
%token I_STV
%token I_STVR
%token I_STVS
%token I_LDP
%token I_LDPT
%token I_STP
%token I_STPR
%token I_STPS
%token I_TRAV
%token I_TRAN
%token I_TRAL
%token I_IPOP
%token I_XPOP
%token I_GENA
%token I_GEND
%token I_GENR
%token I_GEOR
%token I_JMP
%token I_IFT
%token I_IFF
%token I_BOOL
%token I_EQ
%token I_NEQ
%token I_GT
%token I_GE
%token I_LT
%token I_LE
%token I_UNPK
%token I_UNPS
%token I_CALL
%token I_INST
%token I_SWCH
%token I_SELE
%token I_NOP
%token I_TRY
%token I_JTRY
%token I_PTRY
%token I_RIS
%token I_LDRF
%token I_ONCE
%token I_BAND
%token I_BOR
%token I_BXOR
%token I_BNOT
%token I_MODS
%token I_AND
%token I_OR
%token I_ANDS
%token I_ORS
%token I_XORS
%token I_NOTS
%token I_HAS
%token I_HASN
%token I_GIVE
%token I_GIVN
%token I_IN
%token I_NOIN
%token I_PROV
%token I_END
%token I_PEEK
%token I_PSIN
%token I_PASS
%token I_SHR
%token I_SHL
%token I_SHRS
%token I_SHLS
%token I_LDVR
%token I_LDPR
%token I_LSB
%token I_INDI
%token I_STEX
%token I_TRAC
%token I_WRT
%token I_STO
%token I_FORB
%token I_EVAL
%%

/****************************************************
* Rules for falcon.
*****************************************************/


input:
   /* nothing */
   | input line
;

line:
   EOL
   | directive EOL
   | label instruction EOL
   | label EOL
   | instruction EOL
   | error EOL { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
;

xoperand: NIL | operand;
operand: op_variable | op_immediate;
op_variable: SYMBOL | op_register;
op_register: REG_A | REG_B | REG_S1 | REG_S2 | REG_L1 | REG_L2;
//op_imm_simm: op_immediate | SYMBOL;
x_op_immediate: NIL | op_immediate;
op_immediate: NUMERIC | TRUE_TOKEN | FALSE_TOKEN | op_scalar;
op_scalar: STRING_ID | STRING | INTEGER;
op_string: STRING_ID | STRING;
string_or_name: STRING | NAME;


directive:
     DENTRY { COMPILER->addEntry(); }
   | DMODULE NAME { COMPILER->setModuleName( $2 ); }
   | DGLOBAL NAME def_line { COMPILER->addGlobal( $2, $3 ); }
   | DGLOBAL NAME def_line EXPORT { COMPILER->addGlobal( $2, $3, true ); }
   | DVAR NAME x_op_immediate def_line { COMPILER->addVar( $2, $3, $4 ); }
   | DVAR NAME x_op_immediate def_line EXPORT { COMPILER->addVar( $2, $3, $4, true ); }
   | DCONST NAME x_op_immediate { COMPILER->addConst( $2, $3 ); }
   | DCONST NAME x_op_immediate EXPORT { COMPILER->addConst( $2, $3, true ); }
   | DATTRIB NAME def_line { COMPILER->addAttrib( $2, $3 ); }
   | DATTRIB NAME def_line EXPORT { COMPILER->addAttrib( $2, $3, true ); }
   | DLOCAL NAME def_line {  COMPILER->addLocal( $2, $3 ); }
   | DPARAM NAME def_line {  COMPILER->addParam( $2, $3 ); }
   | DFUNCDEF NAME {  COMPILER->addFuncDef( $2 ); }
   | DFUNCDEF NAME EXPORT {  COMPILER->addFuncDef( $2, true ); }
   | DFUNC NAME def_line {  COMPILER->addFunction( $2, $3 ); }
   | DFUNC NAME def_line EXPORT {  COMPILER->addFunction( $2, $3, true ); }
   | DENDFUNC { COMPILER->addFuncEnd(); }
   | DLOAD NAME {  COMPILER->addLoad( $2, false ); }
   | DLOAD STRING {  COMPILER->addLoad( $2, true ); }
   | DIMPORT NAME def_line { COMPILER->addImport( $2, $3 ); }
   | DIMPORT NAME def_line NAME { COMPILER->addImport( $2, $3, $4, 0, false ); }
   | DIMPORT NAME def_line STRING { COMPILER->addImport( $2, $3, $4, 0, true ); }
   | DIMPORT NAME def_line NAME string_or_name {
      COMPILER->addImport( $2, $3, $4, $5, false );
   }
   | DIMPORT NAME def_line STRING string_or_name {
      COMPILER->addImport( $2, $3, $4, $5, true );
   }
   | DALIAS NAME def_line STRING string_or_name {
      COMPILER->addAlias( $2, $3, $4, $5, true );
   }
   | DALIAS NAME def_line NAME string_or_name {
      COMPILER->addAlias( $2, $3, $4, $5, false );
   }
   | DSWITCH op_variable COMMA NAME { COMPILER->addDSwitch( $2, $4 ); }
   | DSWITCH op_variable COMMA INTEGER { COMPILER->addDSwitch( $2, $4 ); }
   | DSELECT op_variable COMMA NAME { COMPILER->addDSwitch( $2, $4, true ); }
   | DSELECT op_variable COMMA INTEGER { COMPILER->addDSwitch( $2, $4, true ); }
   | DCASE NIL COMMA NAME { COMPILER->addDCase( $2, $4 ); }
   | DCASE INTEGER COMMA NAME { COMPILER->addDCase( $2, $4 ); }
   | DCASE STRING COMMA NAME { COMPILER->addDCase( $2, $4 ); }
   | DCASE STRING_ID COMMA NAME { COMPILER->addDCase( $2, $4 ); }
   | DCASE SYMBOL COMMA NAME { COMPILER->addDCase( $2, $4 ); }
   | DCASE INTEGER COLON INTEGER COMMA NAME { COMPILER->addDCase( $2, $6, $4 ); }
   | DENDSWITCH { COMPILER->addDEndSwitch(); }
   | DPROP NAME x_op_immediate {  COMPILER->addProperty( $2, $3 ); }
   | DPROP NAME SYMBOL {  COMPILER->addProperty( $2, $3 ); }
   | DPROPREF NAME SYMBOL {  COMPILER->addPropRef( $2, $3 ); }
   | DHAS has_symlist
   | DHASNT hasnt_symlist
   | DINSTANCE SYMBOL NAME def_line { COMPILER->addInstance( $2, $3, $4 ); }
   | DINSTANCE SYMBOL NAME def_line EXPORT { COMPILER->addInstance( $2, $3, $4, true ); }
   | DCLASS NAME def_line {  COMPILER->addClass( $2, $3 ); }
   | DCLASS NAME def_line EXPORT {  COMPILER->addClass( $2, $3, true ); }
   | DCLASSDEF NAME {  COMPILER->addClassDef( $2 ); }
   | DCLASSDEF NAME EXPORT{  COMPILER->addClassDef( $2, true ); }
   | DCTOR SYMBOL {  COMPILER->addClassCtor( $2 ); }
   | DENDCLASS { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
   | DINHERIT SYMBOL { COMPILER->addInherit($2); }
   | DFROM NAME {  COMPILER->addFrom( $2 ); }
   | DEXTERN NAME def_line {  COMPILER->addExtern( $2, $3 ); }
   | DLINE INTEGER {  COMPILER->addDLine( $2 ); }
   | DSTRING STRING
      {
         // string already added to the module by the lexer
         delete $2;
      }
   | DISTRING STRING
      {
         // string already added to the module by the lexer
         $2->asString().exported( true );
         delete $2;
      }
   | DCSTRING STRING
      {
         // string already added to the module by the lexer
         delete $2;
      }
;


has_symlist:
   SYMBOL { COMPILER->classHas( $1 ); }
   | has_symlist COMMA SYMBOL { COMPILER->classHas( $3 ); }
;

hasnt_symlist:
   SYMBOL { COMPILER->classHasnt( $1 ); }
   | hasnt_symlist COMMA SYMBOL { COMPILER->classHasnt( $3 ); }
;

label: LABEL COLON { COMPILER->defineLabel( $1->asLabel() ); };


def_line:
   /* nothing */ {$$ = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
   | INTEGER
;

instruction:
   inst_ld
   | inst_ldnil
   | inst_add
   | inst_sub
   | inst_mul
   | inst_div
   | inst_mod
   | inst_pow
   | inst_adds
   | inst_subs
   | inst_muls
   | inst_divs
   | inst_inc
   | inst_dec
   | inst_incp
   | inst_decp
   | inst_eq
   | inst_ne
   | inst_ge
   | inst_gt
   | inst_le
   | inst_lt
   | inst_neg
   | inst_not
   | inst_push
   | inst_pshr
   | inst_pop
   | inst_xpop
   | inst_ldv
   | inst_ldvt
   | inst_stv
   | inst_stvr
   | inst_stvs
   | inst_ldp
   | inst_ldpt
   | inst_stp
   | inst_stps
   | inst_stpr
   | inst_trav
   | inst_tran
   | inst_tral
   | inst_ipop
   | inst_gena
   | inst_gend
   | inst_genr
   | inst_geor
   | inst_jmp
   | inst_bool
   | inst_ift
   | inst_iff
   | inst_fork
   | inst_unpk
   | inst_unps
   | inst_call
   | inst_inst
   | inst_ret
   | inst_retval
   | inst_reta
   | inst_nop
   | inst_try
   | inst_ptry
   | inst_jtry
   | inst_ris
   | inst_swch
   | inst_sele
   | inst_ldrf
   | inst_end
   | inst_band
   | inst_bor
   | inst_bxor
   | inst_bnot
   | inst_and
   | inst_or
   | inst_ands
   | inst_ors
   | inst_xors
   | inst_nots
   | inst_mods
   | inst_pows
   | inst_has
   | inst_hasn
   | inst_give
   | inst_givn
   | inst_in
   | inst_noin
   | inst_prov
   | inst_once
   | inst_peek
   | inst_psin
   | inst_pass
   | inst_shr
   | inst_shl
   | inst_shrs
   | inst_shls
   | inst_ldvr
   | inst_ldpr
   | inst_lsb
   | inst_indi
   | inst_stex
   | inst_trac
   | inst_wrt
   | inst_sto
   | inst_forb
   | inst_eval
;

inst_ld:
       I_LD op_variable COMMA xoperand    { COMPILER->addInstr( P_LD, $2, $4 ); }
     | I_LD error              { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
;

inst_ldrf:
       I_LDRF op_variable COMMA xoperand    { COMPILER->addInstr( P_LDRF, $2, $4 ); }
     | I_LDRF error              { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
;

inst_ldnil:
       I_LNIL op_variable { COMPILER->addInstr( P_LNIL, $2 ); }
     | I_LNIL error     { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
;

inst_add:
       I_ADD xoperand COMMA xoperand   { COMPILER->addInstr( P_ADD, $2, $4); }
     | I_ADD error { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
;

inst_adds:
       I_ADDS op_variable COMMA xoperand  { COMPILER->addInstr( P_ADDS, $2, $4); }
     | I_ADDS error { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
;


inst_sub:
       I_SUB xoperand COMMA xoperand   { COMPILER->addInstr( P_SUB, $2, $4); }
     | I_SUB error { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
;

inst_subs:
       I_SUBS op_variable COMMA xoperand  { COMPILER->addInstr( P_SUBS, $2, $4); }
     | I_SUBS error { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
;

inst_mul:
       I_MUL xoperand COMMA xoperand   { COMPILER->addInstr( P_MUL, $2, $4); }
     | I_MUL error { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
;

inst_muls:
       I_MULS op_variable COMMA xoperand { COMPILER->addInstr( P_MULS, $2, $4); }
     | I_MULS error { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
;


inst_div:
       I_DIV xoperand COMMA xoperand    { COMPILER->addInstr( P_DIV, $2, $4); }
     | I_DIV error { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
;

inst_divs:
       I_DIVS op_variable COMMA xoperand   { COMPILER->addInstr( P_DIVS, $2, $4); }
     | I_DIVS error { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
;

inst_mod:
       I_MOD xoperand COMMA xoperand   { COMPILER->addInstr( P_MOD, $2, $4); }
     | I_MOD error { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
;

inst_pow:
       I_POW xoperand COMMA xoperand   { COMPILER->addInstr( P_POW, $2, $4); }
     | I_POW error { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
;


inst_eq:
       I_EQ xoperand COMMA xoperand { COMPILER->addInstr( P_EQ, $2, $4); }
     | I_EQ error { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
;

inst_ne:
       I_NEQ xoperand COMMA xoperand { COMPILER->addInstr( P_NEQ, $2, $4); }
     | I_NEQ error { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
;

inst_ge:
       I_GE xoperand COMMA xoperand { COMPILER->addInstr( P_GE, $2, $4); }
     | I_GE error { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
;

inst_gt:
       I_GT xoperand COMMA xoperand { COMPILER->addInstr( P_GT, $2, $4); }
     | I_GT error { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
;

inst_le:
       I_LE xoperand COMMA xoperand     { COMPILER->addInstr( P_LE, $2, $4); }
     | I_LE error { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
;

inst_lt:
       I_LT xoperand COMMA xoperand { COMPILER->addInstr( P_LT, $2, $4); }
     | I_LT error { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
;

inst_try:
       I_TRY NAME                  { $2->fixed(true); COMPILER->addInstr( P_TRY, $2); }
     | I_TRY INTEGER               { $2->fixed(true); COMPILER->addInstr( P_TRY, $2); }
     | I_TRY error                 { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
;

inst_inc:
       I_INC op_variable           { COMPILER->addInstr( P_INC, $2 ); }
     | I_INC error                 { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
;

inst_dec:
       I_DEC op_variable           { COMPILER->addInstr( P_DEC, $2  ); }
     | I_DEC error                 { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
;


inst_incp:
       I_INCP op_variable           { COMPILER->addInstr( P_INCP, $2 ); }
     | I_INCP error                 { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
;

inst_decp:
       I_DECP op_variable           { COMPILER->addInstr( P_DECP, $2  ); }
     | I_DECP error                 { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
;


inst_neg:
       I_NEG xoperand              { COMPILER->addInstr( P_NEG, $2  ); }
     | I_NEG error                 { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
;

inst_not:
       I_NOT xoperand              { COMPILER->addInstr( P_NOT, $2  ); }
     | I_NOT error                 { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
;

inst_call:
       I_CALL INTEGER COMMA op_variable { $2->fixed( true ); COMPILER->addInstr( P_CALL, $2, $4 ); }
     | I_CALL INTEGER COMMA NAME        { $2->fixed( true ); COMPILER->addInstr( P_CALL, $2, $4 ); }
     | I_CALL error                   { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
;

inst_inst:
       I_INST INTEGER COMMA op_variable { $2->fixed( true ); COMPILER->addInstr( P_INST, $2, $4 ); }
     | I_INST INTEGER COMMA NAME        { $2->fixed( true ); COMPILER->addInstr( P_INST, $2, $4 ); }
     | I_INST error                   { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
;

inst_unpk:
       I_UNPK op_variable COMMA op_variable    { COMPILER->addInstr( P_UNPK, $2, $4 ); }
     | I_UNPK error     { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
;

inst_unps:
       I_UNPS INTEGER COMMA op_variable    { $2->fixed( true ); COMPILER->addInstr( P_UNPS, $2, $4 ); }
     | I_UNPS error     { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
;


inst_push:
       I_PUSH xoperand {  COMPILER->addInstr( P_PUSH, $2 ); }
     | I_PSHN       { COMPILER->addInstr( P_PSHN ); }
     | I_PUSH error { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
;

inst_pshr:
       I_PSHR xoperand { COMPILER->addInstr( P_PSHR, $2 ); }
     | I_PSHR error { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
;


inst_pop:
       I_POP op_variable {  COMPILER->addInstr( P_POP, $2 ); }
     | I_POP error       { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
;

inst_peek:
       I_PEEK op_variable {  COMPILER->addInstr( P_PEEK, $2 ); }
     | I_PEEK error       { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
;

inst_xpop:
       I_XPOP op_variable { COMPILER->addInstr( P_XPOP, $2 ); }
     | I_XPOP error       { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
;


inst_ldv:
       I_LDV op_variable COMMA op_variable { COMPILER->addInstr( P_LDV, $2, $4); }
     | I_LDV op_variable COMMA op_scalar   { COMPILER->addInstr( P_LDV, $2, $4); }
     | I_LDV error              { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
;

inst_ldvt:
       I_LDVT op_variable COMMA op_variable COMMA op_variable { COMPILER->addInstr( P_LDVT, $2, $4, $6); }
     | I_LDVT op_variable COMMA op_scalar COMMA op_variable { COMPILER->addInstr( P_LDVT, $2, $4, $6); }
     | I_LDVT error              { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
;

inst_stv:
       I_STV op_variable COMMA op_variable COMMA xoperand { COMPILER->addInstr( P_STV, $2, $4, $6); }
     | I_STV op_variable COMMA op_scalar  COMMA xoperand  { COMPILER->addInstr( P_STV, $2, $4, $6); }
     | I_STV error              { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
;

inst_stvr:
       I_STVR op_variable COMMA op_variable COMMA xoperand { COMPILER->addInstr( P_STVR, $2, $4, $6); }
     | I_STVR op_variable COMMA op_scalar  COMMA xoperand  { COMPILER->addInstr( P_STVR, $2, $4, $6); }
     | I_STVR error              { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
;

inst_stvs:
       I_STVS op_variable COMMA op_variable  { COMPILER->addInstr( P_STVS, $2, $4 ); }
     | I_STVS op_variable COMMA op_scalar    { COMPILER->addInstr( P_STVS, $2, $4 ); }
     | I_STVS error              { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
;

inst_ldp:
       I_LDP op_variable COMMA op_variable { COMPILER->addInstr( P_LDP, $2, $4); }
     | I_LDP op_variable COMMA op_scalar   { COMPILER->addInstr( P_LDP, $2, $4); }
     | I_LDP error              { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
;

inst_ldpt:
       I_LDPT op_variable COMMA op_variable COMMA op_variable { COMPILER->addInstr( P_LDPT, $2, $4, $6); }
     | I_LDPT op_variable COMMA op_scalar COMMA op_variable   { COMPILER->addInstr( P_LDPT, $2, $4, $6); }
     | I_LDPT error              { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
;

inst_stp:
       I_STP op_variable COMMA op_variable COMMA xoperand { COMPILER->addInstr( P_STP, $2, $4, $6 ); }
     | I_STP op_variable COMMA op_scalar COMMA xoperand   { COMPILER->addInstr( P_STP, $2, $4, $6 ); }
     | I_STP error              { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
;

inst_stpr:
       I_STPR op_variable COMMA op_variable COMMA op_variable { COMPILER->addInstr( P_STPR, $2, $4, $6 ); }
     | I_STPR op_variable COMMA op_scalar COMMA op_variable   { COMPILER->addInstr( P_STPR, $2, $4, $6 ); }
     | I_STPR error              { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
;

inst_stps:
       I_STPS op_variable COMMA op_variable  { COMPILER->addInstr( P_STPS, $2, $4 ); }
     | I_STPS op_variable COMMA op_scalar    { COMPILER->addInstr( P_STPS, $2, $4 ); }
     | I_STPS error              { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
;

inst_trav:
       I_TRAV INTEGER COMMA xoperand COMMA xoperand { $2->fixed( true ); COMPILER->addInstr( P_TRAV, $2, $4, $6); }
     | I_TRAV NAME COMMA xoperand COMMA xoperand    { $2->fixed( true ); COMPILER->addInstr( P_TRAV, $2, $4, $6); }
     | I_TRAV error                   { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
;

inst_tran:
     I_TRAN NAME COMMA NAME COMMA INTEGER { $2->fixed( true ); $4->fixed( true ); $6->fixed( true ); COMPILER->addInstr( P_TRAN, $2, $4, $6 ); }
   | I_TRAN INTEGER COMMA INTEGER COMMA INTEGER { $2->fixed( true ); $4->fixed( true ); $6->fixed( true ); COMPILER->addInstr( P_TRAN, $2, $4, $6 ); }
   | I_TRAN error                   { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
;

inst_tral:
       I_TRAL INTEGER                 { $2->fixed( true ); COMPILER->addInstr( P_TRAL, $2 ); }
     | I_TRAL NAME                    { $2->fixed( true ); COMPILER->addInstr( P_TRAL, $2 ); }
     | I_TRAL error                   { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
;

inst_ipop:
       I_IPOP  INTEGER                { $2->fixed( true ); COMPILER->addInstr( P_IPOP, $2 ); }
     | I_IPOP error                   { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
;

inst_gena:
      I_GENA INTEGER            { $2->fixed( true ); COMPILER->addInstr( P_GENA, $2 ); }
   |  I_GENA error              { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
;

inst_gend:
      I_GEND INTEGER            { $2->fixed( true ); COMPILER->addInstr( P_GEND, $2 ); }
   |  I_GEND error              { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
;

inst_genr:
       I_GENR operand COMMA operand COMMA x_op_immediate { COMPILER->addInstr( P_GENR, $2, $4, $6); }
     | I_GENR error               { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
;

inst_geor:
       I_GEOR operand             { COMPILER->addInstr( P_GEOR, $2 ); }
     | I_GEOR error               { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
;

inst_ris:
      I_RIS  xoperand           { COMPILER->addInstr( P_RIS, $2 ); }
   |  I_RIS  error              { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
;

inst_jmp:
      I_JMP NAME               { $2->fixed( true ); COMPILER->addInstr( P_JMP, $2 ); }
   |  I_JMP INTEGER            { $2->fixed( true ); COMPILER->addInstr( P_JMP, $2 ); }
   |  I_JMP error              { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
;

inst_bool:
      I_BOOL xoperand          { COMPILER->addInstr( P_BOOL, $1 ); }
   |  I_BOOL error             { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
;

inst_ift:
      I_IFT NAME COMMA xoperand      { $2->fixed( true ); COMPILER->addInstr( P_IFT, $2, $4 ); }
   |  I_IFT INTEGER COMMA xoperand   { $2->fixed( true ); COMPILER->addInstr( P_IFT, $2, $4 ); }
   |  I_IFT error              { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
;

inst_iff:
      I_IFF NAME COMMA xoperand     { $2->fixed( true ); COMPILER->addInstr( P_IFF, $2, $4 ); }
   |  I_IFF INTEGER COMMA xoperand  { $2->fixed( true ); COMPILER->addInstr( P_IFF, $2, $4 ); }
   |  I_IFF error              { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
;


inst_fork:
      I_FORK INTEGER COMMA NAME      { $2->fixed( true ); $4->fixed( true ); COMPILER->addInstr( P_FORK, $2, $4 ); }
   |  I_FORK INTEGER COMMA INTEGER   { $2->fixed( true ); $4->fixed( true ); COMPILER->addInstr( P_FORK, $2, $4 ); }
   |  I_FORK error              { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
;

inst_jtry:
      I_JTRY NAME               { $2->fixed( true ); COMPILER->addInstr( P_JTRY, $2 ); }
   |  I_JTRY INTEGER            { $2->fixed( true ); COMPILER->addInstr( P_JTRY, $2 ); }
   |  I_JTRY error              { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
;

inst_ret:
     I_RET       { COMPILER->addInstr( P_RET ); }
   | I_RET error { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
;

inst_reta:
     I_RETA       { COMPILER->addInstr( P_RETA ); }
   | I_RETA error { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
;

inst_retval:
     I_RETV xoperand  { COMPILER->addInstr( P_RETV, $2 ); }
   | I_RETV error { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
;

inst_nop:
     I_NOP           { COMPILER->addInstr( P_NOP ); }
   | I_NOP error     { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
;

inst_ptry:
     I_PTRY INTEGER   { $2->fixed( true ); COMPILER->addInstr( P_PTRY, $2 ); }
   | I_PTRY error     { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
;

inst_end:
     I_END       { COMPILER->addInstr( P_END ); }
   | I_END error     { COMPILER->raiseError(Falcon::e_invop, "END" ); }
;

inst_swch:
      I_SWCH INTEGER COMMA op_variable COMMA switch_list { $2->fixed(true); COMPILER->write_switch( $2, $4, $6 ); }
   |  I_SWCH error                                 { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
;

inst_sele:
      I_SELE INTEGER COMMA op_variable COMMA switch_list { $2->fixed(true); COMPILER->write_switch( $2, $4, $6 ); }
   |  I_SELE error                                 { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
;

switch_list:
      INTEGER COMMA NAME
      {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( $1 );
         psd->asList()->pushBack( $3 );
         $$ = psd;
      }

   |  switch_list COMMA INTEGER COMMA NAME
      {
         $1->asList()->pushBack( $3 );
         $1->asList()->pushBack( $5 );
         $$ = $1;
      }
;

inst_once:
      I_ONCE INTEGER COMMA xoperand  { $2->fixed( true ); COMPILER->addInstr( P_ONCE, $2, $4 ); COMPILER->addStatic(); }
   |  I_ONCE NAME COMMA xoperand       { $2->fixed( true ); COMPILER->addInstr( P_ONCE, $2, $4 ); COMPILER->addStatic(); }
   |  I_ONCE error                 { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
;

inst_band:
      I_BAND SYMBOL COMMA SYMBOL     { COMPILER->addInstr( P_BAND, $2, $4 ); }
   |  I_BAND SYMBOL COMMA INTEGER    { COMPILER->addInstr( P_BAND, $2, $4 ); }
   |  I_BAND INTEGER COMMA SYMBOL    { COMPILER->addInstr( P_BAND, $2, $4 ); }
   |  I_BAND INTEGER COMMA INTEGER   { COMPILER->addInstr( P_BAND, $2, $4 ); }
   |  I_BAND error                 { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
;

inst_bor:
      I_BOR SYMBOL COMMA SYMBOL     { COMPILER->addInstr( P_BOR, $2, $4 ); }
   |  I_BOR SYMBOL COMMA INTEGER    { COMPILER->addInstr( P_BOR, $2, $4 ); }
   |  I_BOR INTEGER COMMA SYMBOL    { COMPILER->addInstr( P_BOR, $2, $4 ); }
   |  I_BOR INTEGER COMMA INTEGER   { COMPILER->addInstr( P_BOR, $2, $4 ); }
   |  I_BOR error                 { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
;

inst_bxor:
      I_BXOR SYMBOL COMMA SYMBOL     { COMPILER->addInstr( P_BXOR, $2, $4 ); }
   |  I_BXOR SYMBOL COMMA INTEGER    { COMPILER->addInstr( P_BXOR, $2, $4 ); }
   |  I_BXOR INTEGER COMMA SYMBOL    { COMPILER->addInstr( P_BXOR, $2, $4 ); }
   |  I_BXOR INTEGER COMMA INTEGER   { COMPILER->addInstr( P_BXOR, $2, $4 ); }
   |  I_BXOR error                 { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
;

inst_bnot:
      I_BNOT SYMBOL                { COMPILER->addInstr( P_BNOT, $2 ); }
   |  I_BNOT INTEGER               { COMPILER->addInstr( P_BNOT, $2 ); }
   |  I_BNOT error                 { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
;

inst_and:
       I_AND  xoperand COMMA xoperand   { COMPILER->addInstr( P_AND, $2, $4); }
     | I_AND error { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
;

inst_or:
       I_OR xoperand COMMA xoperand   { COMPILER->addInstr( P_OR, $2, $4); }
     | I_OR error { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
;

inst_ands:
       I_ANDS op_variable COMMA operand   { COMPILER->addInstr( P_ANDS, $2, $4); }
     | I_ANDS error { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
;

inst_ors:
       I_ORS op_variable COMMA operand   { COMPILER->addInstr( P_ORS, $2, $4); }
     | I_ORS error { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
;

inst_xors:
       I_XORS op_variable COMMA operand   { COMPILER->addInstr( P_XORS, $2, $4); }
     | I_XORS error { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
;

inst_mods:
       I_MODS op_variable COMMA operand   { COMPILER->addInstr( P_MODS, $2, $4); }
     | I_MODS error { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
;

inst_pows:
       I_POWS op_variable COMMA operand   { COMPILER->addInstr( P_POWS, $2, $4); }
     | I_POWS error { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
;

inst_nots:
       I_NOTS op_variable   { COMPILER->addInstr( P_NOTS, $2 ); }
     | I_NOTS error         { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
;

inst_has:
       I_HAS  op_variable COMMA op_variable   { COMPILER->addInstr( P_HAS, $2, $4); }
     | I_HAS  op_variable COMMA INTEGER       { COMPILER->addInstr( P_HAS, $2, $4); }
     | I_HAS error { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
;

inst_hasn:
       I_HASN op_variable COMMA op_variable   { COMPILER->addInstr( P_HASN, $2, $4); }
     | I_HASN op_variable COMMA INTEGER       { COMPILER->addInstr( P_HASN, $2, $4); }
     | I_HASN error { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
;

inst_give:
       I_GIVE op_variable COMMA op_variable   { COMPILER->addInstr( P_GIVE, $2, $4); }
     | I_GIVE op_variable COMMA INTEGER       { COMPILER->addInstr( P_GIVE, $2, $4); }
     | I_GIVE error { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
;

inst_givn:
       I_GIVN op_variable COMMA op_variable   { COMPILER->addInstr( P_GIVN, $2, $4); }
     | I_GIVN op_variable COMMA INTEGER       { COMPILER->addInstr( P_GIVN, $2, $4); }
     | I_GIVN error { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
;


inst_in:
       I_IN xoperand COMMA operand   { COMPILER->addInstr( P_IN, $2, $4); }
     | I_IN error                  { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
;

inst_noin:
       I_NOIN xoperand COMMA operand   { COMPILER->addInstr( P_NOIN, $2, $4); }
     | I_NOIN error                  { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
;

inst_prov:
       I_PROV xoperand COMMA operand   { COMPILER->addInstr( P_PROV, $2, $4); }
     | I_PROV error                  { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
;

inst_psin:
       I_PSIN op_variable        { COMPILER->addInstr( P_PSIN, $2 ); }
     | I_PSIN error              { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
;

inst_pass:
       I_PASS op_variable        { COMPILER->addInstr( P_PASS, $2 ); }
     | I_PASS error              { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
;

inst_shr:
       I_SHR xoperand COMMA xoperand    { COMPILER->addInstr( P_SHR, $2, $4 ); }
     | I_SHR error              { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
;

inst_shl:
       I_SHL xoperand COMMA xoperand    { COMPILER->addInstr( P_SHL, $2, $4 ); }
     | I_SHL error              { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
;

inst_shrs:
       I_SHRS xoperand COMMA xoperand    { COMPILER->addInstr( P_SHRS, $2, $4 ); }
     | I_SHRS error              { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
;

inst_shls:
       I_SHLS xoperand COMMA xoperand    { COMPILER->addInstr( P_SHLS, $2, $4 ); }
     | I_SHLS error              { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
;

inst_ldvr:
       I_LDVR xoperand COMMA xoperand    { COMPILER->addInstr( P_LDVR, $2, $4 ); }
     | I_LDVR error              { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
;

inst_ldpr:
       I_LDPR xoperand COMMA xoperand    { COMPILER->addInstr( P_LDPR, $2, $4 ); }
     | I_LDPR error              { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
;

inst_lsb:
       I_LSB xoperand COMMA xoperand    { COMPILER->addInstr( P_LSB, $2, $4 ); }
     | I_LSB error              { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
;

inst_indi:
       I_INDI op_string          { COMPILER->addInstr( P_INDI, $2 ); }
     | I_INDI op_variable        { COMPILER->addInstr( P_INDI, $2 ); }
     | I_INDI error              { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
;

inst_stex:
       I_STEX op_string          { COMPILER->addInstr( P_STEX, $2 ); }
     | I_STEX op_variable        { COMPILER->addInstr( P_STEX, $2 ); }
     | I_STEX error              { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
;

inst_trac:
       I_TRAC xoperand           { COMPILER->addInstr( P_TRAC, $2 ); }
     | I_TRAC error              { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
;

inst_wrt:
       I_WRT xoperand           { COMPILER->addInstr( P_WRT, $2 ); }
     | I_WRT error              { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
;


inst_sto:
       I_STO op_variable COMMA xoperand    { COMPILER->addInstr( P_STO, $2, $4 ); }
     | I_STO error              { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
;

inst_forb:
       I_FORB op_string COMMA xoperand    { COMPILER->addInstr( P_FORB, $2, $4 ); }
     | I_FORB error              { COMPILER->raiseError(Falcon::e_invop, "FORB" ); }
;

inst_eval:
       I_EVAL xoperand     { COMPILER->addInstr( P_EVAL, $2 ); }
     | I_EVAL error        { COMPILER->raiseError(Falcon::e_invop, "EVAL" ); }
;

%% /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */

