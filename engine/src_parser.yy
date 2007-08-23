/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.ypp
   $Id: src_parser.yy,v 1.43 2007/08/01 13:24:55 jonnymind Exp $

   Bison grammar definition for falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven mag 21 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/


%{

#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__

%}



%union lex_value_t {
   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
}

/*%debug*/
%pure_parser
%defines
%name-prefix="flc_src_"


%token EOL
%token <integer> INTNUM
%token <numeric> DBLNUM
%token <stringp> SYMBOL
%token <stringp> STRING

%token NIL
%token END
%token DEF
%token WHILE BREAK CONTINUE DROPPING
%token IF ELSE ELIF
%token FOR
%token FORFIRST FORLAST FORALL
%token SWITCH CASE DEFAULT
%token SELECT
%token SENDER SELF
%token GIVE
%token TRY CATCH RAISE
%token CLASS FROM OBJECT
%token RETURN
%token GLOBAL
%token LAMBDA
%token INIT
%token LOAD
%token LAUNCH
%token CONST_KW
%token ATTRIBUTES
%token PASS
%token EXPORT
%token COLON
%token FUNCDECL STATIC
%token FORDOT
%token LOOP

%token CLOSEPAR OPENPAR CLOSESQUARE OPENSQUARE DOT
/*
   Assigning rule precendence: immediate operations have maximum precedence, being resolved immediately,
   then the assignment gets more precendece than the expressions (OP_EQ is preferibily parsed as an
   assignment request where ambiguity arises).
*/

%right ASSIGN_ADD ASSIGN_SUB ASSIGN_MUL ASSIGN_DIV ASSIGN_MOD ASSIGN_BAND ASSIGN_BOR ASSIGN_BXOR ASSIGN_SHR ASSIGN_SHL ASSIGN_POW
%left ARROW
%right COMMA OP_TO FOR_STEP
%left QUESTION
%right COLON
%left OR
%left AND
%right NOT
%right LET
%left OP_ASSIGN OP_EQ EEQ NEQ GT LT GE LE
%left HAS HASNT OP_IN OP_NOTIN PROVIDES
%right ATSIGN DIESIS
%left VBAR CAP
%left AMPER
%left PLUS MINUS
%left STAR SLASH PERCENT
%left POW
%left SHL SHR
%right NEG BANG
%right DOLLAR INCREMENT DECREMENT

%type <fal_adecl> expression_list par_expression_list
%type <fal_adecl> symbol_list inherit_param_list inherit_call
%type <fal_ddecl> expression_pair_list
%type <fal_val> expression variable func_call lambda_expr iif_expr
%type <fal_val> switch_decl select_decl while_decl while_short_decl
%type <fal_val> if_decl if_short_decl elif_decl
%type <fal_val> const_atom var_atom  atomic_symbol /* atom */
%type <stringp> load_statement
%type <fal_val> array_decl dict_decl
%type <fal_val> inherit_param_token
%type <fal_stat> break_statement continue_statement
%type <fal_stat> toplevel_statement statement assignment while_statement if_statement
%type <fal_stat> for_statement for_decl for_decl_short forin_statement switch_statement fordot_statement
%type <fal_stat> select_statement
%type <fal_stat> give_statement try_statement raise_statement return_statement global_statement
%type <fal_stat> base_statement
%type <fal_stat> launch_statement
%type <fal_stat> pass_statement
%type <fal_stat> func_statement
%type <fal_stat> self_print_statement
%type <fal_stat> class_decl object_decl property_decl attributes_statement export_statement
%type <fal_stat> def_statement

%type <fal_stat> op_assignment autoadd autosub automul autodiv automod autoband autobor autobxor autopow const_statement
%type <fal_stat> autoshl autoshr
%type <fal_val>  range_decl

%%

/****************************************************
* Rules for falcon.
*****************************************************/

/*
input:
   {yydebug = 1; } body
;
*/

input:
   body
;

body:
   /*empty */
   | body line
;

line:
   toplevel_statement
   | END EOL { COMPILER->raiseError(Falcon::e_lone_end ); }
   | CASE error EOL { COMPILER->raiseError(Falcon::e_case_outside ); }

;

toplevel_statement:
   load_statement
      {
         if ( $1 != 0 )
            COMPILER->addLoad( *$1 );
      }
   | func_statement
      {
         if( $1 != 0 )
            COMPILER->addFunction( $1 );
      }
   | class_decl
      {
         if ( $1 != 0 )
            COMPILER->addClass( $1 );
      }
   | object_decl
      {
         if ( $1 != 0 )
            COMPILER->addClass( $1 );
      }
   | statement
      {
         if( $1 != 0 )
            COMPILER->addStatement( $1 );
      }
   | const_statement /* no action */
   | export_statement /* no action */
   | attributes_statement /* no action */
;

load_statement:
   LOAD SYMBOL EOL
      {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         $$ = $2;
      }
   | LOAD STRING EOL
      {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         $$ = $2;
      }
   | LOAD error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_load );
         $$ = 0;
      }
;

statement:
   base_statement { COMPILER->checkLocalUndefined(); $$ = $1; }
   | EOL { $$ = 0; }

   /* Some errors, at this level: */
   | FUNCDECL error EOL { COMPILER->raiseError(Falcon::e_toplevel_func ); $$ = 0; }
   | OBJECT error EOL { COMPILER->raiseError(Falcon::e_toplevel_obj ); $$ = 0; }
   | CLASS error EOL { COMPILER->raiseError(Falcon::e_toplevel_class ); $$ = 0; }
   | error EOL { COMPILER->raiseError(Falcon::e_syntax ); $$ = 0;}

;

assignment_def_list:
   atomic_symbol OP_ASSIGN expression {
      COMPILER->defContext( true ); COMPILER->defRequired();
      COMPILER->defineVal( $1 );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, $1, $3 ) );
   }
   | assignment_def_list COMMA atomic_symbol OP_ASSIGN expression {
      COMPILER->defineVal( $3 );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, $3, $5 ) );
   }
;

base_statement:
   expression EOL { $$ = new Falcon::StmtAutoexpr( LINE, $1 ); }
   | def_statement /* no action  -- at the moment def_statement always returns 0*/
   | assignment
   | op_assignment
   | while_statement
   | for_statement
   | forin_statement
   | switch_statement
   | select_statement
   | if_statement
   | break_statement
   | continue_statement
   | give_statement
   | try_statement
   | raise_statement
   | return_statement
   | global_statement
   | launch_statement
   | pass_statement
   | fordot_statement
   | self_print_statement
;

def_statement:
   DEF assignment_def_list EOL
      { COMPILER->defContext( false );  $$=0; }
   | DEF error EOL
      { COMPILER->raiseError( Falcon::e_syn_def ); }
;


assignment:
   variable OP_ASSIGN expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAssignment( LINE, $1, $3 );
   }
   | variable OP_ASSIGN DOLLAR DOLLAR EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtUnref( LINE, $1 );
   }
   | variable OP_ASSIGN expression_list EOL {
         COMPILER->defineVal( $1 );
         $$ = new Falcon::StmtAssignment( LINE, $1, new Falcon::Value( $3 ) );
      }
   | variable COMMA symbol_list OP_ASSIGN expression EOL {
         COMPILER->defineVal( $1 );
         $3->pushFront( $1 );
         $$ = new Falcon::StmtAssignment( LINE, new Falcon::Value($3), $5 );
      }
   | variable COMMA symbol_list OP_ASSIGN expression_list EOL {
         COMPILER->defineVal( $1 );
         $3->pushFront( $1 );
         $$ = new Falcon::StmtAssignment( LINE, new Falcon::Value($3), new Falcon::Value( $5 ) );
      }
;



op_assignment:
   autoadd
   | autosub
   | automul
   | autodiv
   | automod
   | autopow
   | autoband
   | autobor
   | autobxor
   | autoshl
   | autoshr
;

autoadd:
   expression ASSIGN_ADD expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoAdd( LINE, $1, $3 );
   }
;

autosub:
   variable ASSIGN_SUB expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoSub( LINE, $1, $3 );
   }
;

automul:
   variable ASSIGN_MUL expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoMul( LINE, $1, $3 );
   }
;

autodiv:
   variable ASSIGN_DIV expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoDiv( LINE, $1, $3 );
   }
;

automod:
   variable ASSIGN_MOD expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoMod( LINE, $1, $3 );
   }
;

autopow:
   variable ASSIGN_POW expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoPow( LINE, $1, $3 );
   }
;

autoband:
   variable ASSIGN_BAND expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoBAND( LINE, $1, $3 );
   }
;

autobor:
   variable ASSIGN_BOR expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoBOR( LINE, $1, $3 );
   }
;

autobxor:
   variable ASSIGN_BXOR expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoBXOR( LINE, $1, $3 );
   }
;
autoshl:
   variable ASSIGN_SHL expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoSHL( LINE, $1, $3 );
   }
;
autoshr:
   variable ASSIGN_SHR expression EOL {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::StmtAutoSHR( LINE, $1, $3 );
   }
;


while_statement:
   while_decl {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, $1 );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
      statement_list END EOL
      {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = w;
      }

   | while_short_decl statement {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, $1 );
         if ( $2 != 0 )
            w->children().push_back( $2 );
         $$ = w;
      }

while_decl:
   WHILE expression EOL { $$ = $2; }
   | LOOP EOL { $$ = 0; }
   | WHILE error EOL { COMPILER->raiseError(Falcon::e_syn_while ); $$ = 0; }
;

while_short_decl:
   WHILE expression COLON { $$ = $2; }
   | LOOP COLON { $$ = 0; }
   | WHILE error COLON { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); $$ = 0; }
;

if_statement:
   if_decl {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, $1 );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
      statement_list
      elif_or_else
      END EOL
      {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = stmt;
      }

   | if_short_decl statement {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, $1 );
         if( $2 != 0 )
            stmt->children().push_back( $2 );
         $$ = stmt;
      }
;

if_decl:
   IF expression EOL { $$ = $2; }
   | IF error EOL {  COMPILER->raiseError(Falcon::e_syn_if ); $$ = 0; }
;

if_short_decl:
   IF expression COLON { $$ = $2; }
   | IF error COLON {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); $$ = 0; }
;


elif_or_else:
   /* nothing */
   | elif_statement
   | else_decl {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
      statement_list
;

else_decl:
   ELSE EOL
   | ELSE error EOL { COMPILER->raiseError(Falcon::e_syn_else ); }
;


elif_statement:
   elif_decl {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, $1 );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
      statement_list
      elif_or_else
;

elif_decl:
   ELIF expression EOL { $$ = $2; }
   | ELIF error EOL { COMPILER->raiseError(Falcon::e_syn_elif ); $$ = 0; }
;

statement_list:
   /* empty */
   | statement_list statement {
      COMPILER->addStatement( $2 );
   }
;

break_statement:
   BREAK EOL
      {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            $$ = 0;
         }
         else
            $$ = new Falcon::StmtBreak( LINE );
      }
   | BREAK error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_break );
         $$ = 0;
      }
;

continue_statement:
   CONTINUE EOL
      {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            $$ = 0;
         }
         else
            $$ = new Falcon::StmtContinue( LINE );
      }

   | CONTINUE DROPPING EOL
      {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            $$ = 0;
         }
         else
            $$ = new Falcon::StmtContinue( LINE, true );
      }
   | CONTINUE error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_continue );
         $$ = 0;
      }
;

for_statement:
   for_decl {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( $1 );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
      statement_list END EOL
      {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = f;
      }
   | for_decl_short statement
      {
         if ( $1 != 0 )
         {
            Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>($1);
            if ( $2 != 0 )
                f->children().push_back( $2 );
            $$ = f;
         }
         else
            delete $2;
      }
;


for_decl:
   FOR variable OP_ASSIGN expression OP_TO expression EOL{
         COMPILER->defineVal( $2 );
         $$ = new Falcon::StmtFor( LINE, $2, $4, $6 );
      }
   | FOR variable OP_ASSIGN expression OP_TO expression FOR_STEP expression EOL{
         COMPILER->defineVal( $2 );
         $$ = new Falcon::StmtFor( LINE, $2, $4, $6, $8 );
      }
   | FOR error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_for );
         $$ = new Falcon::StmtFor( LINE, 0, 0, 0 );
      }
;

for_decl_short:
   FOR variable OP_ASSIGN expression OP_TO expression COLON{
         COMPILER->defineVal( $2 );
         $$ = new Falcon::StmtFor( CURRENT_LINE, $2, $4, $6 );
      }
   | FOR variable OP_ASSIGN expression OP_TO expression FOR_STEP expression COLON{
         COMPILER->defineVal( $2 );
         $$ = new Falcon::StmtFor( CURRENT_LINE, $2, $4, $6, $8 );
      }
   | FOR error COLON
      {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         $$ = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      }
;


forin_statement:
   FOR symbol_list OP_IN expression EOL
      {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = $2;
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), $4 );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), $4 );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
      forin_statement_list
      END EOL
      {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = f;
      }
   | FOR symbol_list OP_IN expression COLON statement
      {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = $2;
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), $4 );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), $4 );
         if ( $6 != 0 )
             f->children().push_back( $6 );
      }
   | FOR symbol_list OP_IN error EOL
       { COMPILER->raiseError( Falcon::e_syn_forin ); }
;

forin_statement_list:
   /* nothing */
   | forin_statement_elem forin_statement_list
;

forin_statement_elem:
   statement {
         if ( $1 != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( $1 );
         }
      }
   | first_loop_block
   | last_loop_block
   | all_loop_block
;

fordot_statement:
   FORDOT expression EOL
      {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete $2;
            $$ = 0;
         }
         else {
            $$ = new Falcon::StmtFordot( LINE, $2 );
         }
      }
   | FORDOT error EOL
      {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         $$ = 0;
      }
;

self_print_statement:
   LT  expression_list EOL
      {
         $$ = new Falcon::StmtSelfPrint( LINE, $2 );
      }
   | LT EOL
      {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         $$ = 0;
      }

   | GT  expression_list EOL
      {
         $2->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         $$ = new Falcon::StmtSelfPrint( LINE, $2 );
      }
   | GT EOL
      {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         $$ = new Falcon::StmtSelfPrint( LINE, adecl );
      }

   | LT error EOL
      {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         $$ = 0;
      }
;

first_loop_block:
   FORFIRST EOL {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
      statement_list
      END EOL
      { COMPILER->popContextSet(); }

   | FORFIRST COLON statement {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         f->firstBlock().push_back( $3 );
      }
   | FORFIRST error EOL { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
;

last_loop_block:
   FORLAST EOL {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
      statement_list
      END EOL
      { COMPILER->popContextSet(); }
   | FORLAST COLON statement {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         f->lastBlock().push_back( $3 );
      }
   | FORLAST error EOL { COMPILER->raiseError(Falcon::e_syn_forlast ); }
;

all_loop_block:
   FORALL EOL {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->allBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->allBlock() );
      }
      statement_list
      END EOL
      { COMPILER->popContextSet(); }

   | FORALL COLON statement {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         f->allBlock().push_back( $3 );
      }
   | FORALL error EOL { COMPILER->raiseError(Falcon::e_syn_forall ); }
;

switch_statement:
   switch_decl {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, $1 );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
      case_list
      default_statement
      END EOL
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = stmt;
      }
;

switch_decl:
   SWITCH expression EOL { $$ = $2; }
   |SWITCH error EOL
      {
         COMPILER->raiseError(Falcon::e_switch_decl );
         $$ = 0;
      }
;

case_list:
   /* nothing */
   | case_list case_statement
   | case_list error EOL { COMPILER->raiseError(Falcon::e_switch_body ); }
;

case_statement:
   EOL
   | CASE case_expression_list EOL
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement_list

   | CASE case_expression_list COLON {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement {
            COMPILER->addStatement( $5 );
      }

   | CASE error EOL {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement_list

   | CASE error COLON {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement {
            COMPILER->addStatement( $5 );
      }

;

default_statement:
   /* noting */
   | default_decl
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
     default_body
;

default_decl:
   | DEFAULT
   | DEFAULT error { COMPILER->raiseError(Falcon::e_default_decl ); }

default_body:
   EOL  statement_list
   | COLON statement {
         COMPILER->addStatement( $2 );
      }
;

case_expression_list:
   case_element
   | case_expression_list COMMA case_element
;

case_element:
   NIL
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }

   | INTNUM
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( $1 );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }

   | STRING
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( $1 );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }

   | INTNUM OP_TO INTNUM
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( $1 ), new Falcon::Value( $3 ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }

   | SYMBOL
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( $1 );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( $1 );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
;

/*****************************************************
   select statement
******************************************************/

select_statement:
   select_decl {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, $1 );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
      selcase_list
      default_statement
      END EOL
      {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = stmt;
      }
;

select_decl:
   SELECT expression EOL { $$ = $2; }
   |SELECT error EOL
      {
         COMPILER->raiseError(Falcon::e_select_decl );
         $$ = 0;
      }
;

selcase_list:
   /* nothing */
   | selcase_list selcase_statement
   | selcase_list error EOL { COMPILER->raiseError(Falcon::e_select_body ); }
;

selcase_statement:
   EOL
   | CASE selcase_expression_list EOL
      {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement_list

   | CASE selcase_expression_list COLON {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }

      statement {
            COMPILER->addStatement( $5 );
      }

   | CASE error EOL {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement_list

   | CASE error COLON {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
      statement {
            COMPILER->addStatement( $5 );
      }

;
;

selcase_expression_list:
   selcase_element
   | selcase_expression_list COMMA selcase_element
;

selcase_element:
   | INTNUM
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( $1 );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }

   | SYMBOL
      {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( $1 );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( $1 );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
;

/*****************************************************
   give statement
******************************************************/

give_statement:
   GIVE expression_list OP_TO expression EOL
      {
         $$ = new Falcon::StmtGive( LINE, $4, $2 );
      }
   | GIVE expression_list error EOL
      {
         $$ = new Falcon::StmtGive( LINE, 0, $2 );
         COMPILER->raiseError(Falcon::e_syn_give );
      }
   | GIVE error EOL { COMPILER->raiseError(Falcon::e_syn_give ); $$ = 0; }
;

/*****************************************************
   Try statement
******************************************************/

try_statement:
   TRY COLON statement {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( $3 != 0 )
          t->children().push_back( $3 );
      $$ = t;
   }
   | try_decl
      {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }

      statement_list
      catch_statements
      END EOL

      {
         $$ = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
;

try_decl:
   TRY EOL
   | TRY error EOL { COMPILER->raiseError(Falcon::e_syn_try ); }
;

catch_statements:
   /* nothing */
   | catch_list
;

catch_list:
   catch_statement
   | catch_list catch_statement
;

catch_statement:
   catch_decl
   statement_list
;

catch_decl:
   CATCH EOL
      {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }

   | CATCH OP_IN atomic_symbol EOL
      {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( $3 );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, $3 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }

   | CATCH catchcase_element_list EOL
      {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }

   | CATCH catchcase_element_list OP_IN atomic_symbol EOL
      {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( $4 );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, $4 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }

   | CATCH error EOL
   {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }

;

catchcase_element_list:
   catchcase_element
   | catchcase_element_list COMMA catchcase_element;
;

catchcase_element:
   INTNUM
      {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( $1 );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }

   | SYMBOL
      {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( $1 );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( $1 );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
;


/**********************************************************
   RAISE statement
***********************************************************/

raise_statement:
   RAISE expression EOL { $$ = new Falcon::StmtRaise( LINE, $2 ); }
   | RAISE error EOL { COMPILER->raiseError(Falcon::e_syn_raise ); $$ = 0; }
;

/**********************************************************
   Function declaration
***********************************************************/
func_statement:
   func_decl
      static_block
      statement_list

      END EOL
      {
         $$ = COMPILER->getContext();
         COMPILER->closeFunction();
      }

   | func_decl_short statement
      {
         COMPILER->addStatement( $2 );
         $$ = COMPILER->getContext();
         COMPILER->closeFunction();
      }
;

func_decl:
   func_begin OPENPAR opt_eol param_list opt_eol CLOSEPAR EOL
   | func_begin OPENPAR opt_eol param_list error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol CLOSEPAR EOL
      {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      }
   | func_begin error EOL { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
;

func_decl_short:
   func_begin OPENPAR opt_eol param_list opt_eol CLOSEPAR COLON
   | func_begin OPENPAR opt_eol error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol CLOSEPAR COLON
      {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      }
;

func_begin:
   FUNCDECL SYMBOL
      {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *$2;
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = $2;

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
            if ( cd->hasProperty( *$2 ) ) {
               COMPILER->raiseError(Falcon::e_prop_adef, *$2 );
            }
            else {
                cd->addProperty( $2, new Falcon::VarDef( sym ) );
            }
         }

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
;


param_list:
   /* nothing */
   | param_symbol
   | param_list COMMA opt_eol param_symbol
;

param_symbol:
   SYMBOL
      {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( $1 );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), $1 );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      }
;

static_block:
   /* nothing */
   | static_decl
      {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
      statement_list END EOL
      {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
   | static_short_decl
      {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
      statement
      {
         COMPILER->addStatement( $3 );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
;

static_decl:
   STATIC EOL
   | STATIC error EOL { COMPILER->raiseError(Falcon::e_syn_static ); }
;

static_short_decl:
   STATIC COLON
   | STATIC error COLON { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
;


/**********************************************************
   Launch Statement
***********************************************************/

launch_statement:
   LAUNCH func_call EOL
      {
         $$ = new Falcon::StmtLaunch( LINE, $2 );
      }
   | LAUNCH error EOL { COMPILER->raiseError(Falcon::e_syn_launch ); $$ = 0; }
;

/**********************************************************
   Pass Statement
***********************************************************/

pass_statement:
   PASS expression EOL
      {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            $$ = new Falcon::StmtPass( LINE, $2 );
      }
   | PASS expression OP_IN expression EOL
      {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( $4 );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            $$ = new Falcon::StmtPass( LINE, $2, $4 );
      }
   | PASS expression OP_IN error EOL
      {
         delete $2;
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         $$ = 0;
      }
   | PASS error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_pass );
         $$ = 0;
      }
;

/**********************************************************
   Const Statement
***********************************************************/

const_statement:
   CONST_KW SYMBOL OP_ASSIGN const_atom EOL
      {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = $4; //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *$2, val, LINE );
         // we don't need the expression anymore
         // no other action:
         $$ = 0;
      }
   | CONST_KW SYMBOL OP_ASSIGN error EOL
      {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         $$ = 0;
      }
   | CONST_KW error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_const );
         $$ = 0;
      }
;

/**********************************************************
   Export directive
***********************************************************/

export_statement:
   EXPORT EOL
      {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         $$=0;
      }
   | EXPORT export_symbol_list EOL
      {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         $$ = 0;
      }
   | EXPORT error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_export );
         $$ = 0;
      }
;

export_symbol_list:
   SYMBOL
      {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( $1 );
         sym->exported(true);
      }
   | export_symbol_list COMMA SYMBOL
      {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( $3 );
         sym->exported(true);
      }
;

/**********************************************************
   Attributes Statement
***********************************************************/

attributes_statement:
   attributes_decl
      attribute_vert_list
      END EOL
      {
         // no other action:
         $$ = 0;
      }
   | attributes_short_decl
      attribute_list
      EOL
      {
         // no other action:
         $$ = 0;
      }
;

attributes_decl:
   ATTRIBUTES EOL
   | ATTRIBUTES error EOL { COMPILER->raiseError(Falcon::e_syn_attributes ); }

attributes_short_decl:
   ATTRIBUTES COLON
   | ATTRIBUTES error COLON { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); }
;

attribute_list:
   /* nothing */
   | SYMBOL
         {
            COMPILER->addAttribute( $1 );
         }
   | attribute_list COMMA SYMBOL
         {
            COMPILER->addAttribute( $3 );
         }
;

attribute_vert_list:
   attribute_list EOL
   | attribute_vert_list attribute_list EOL
   | error EOL
   {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   }
;

/**********************************************************
   Class Declaration
***********************************************************/

class_decl:
      CLASS SYMBOL
      {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( $2 );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( $2 );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      }
      /* param_list convert the above classdef in a funcdef. */
      class_def_inner

      class_statement_list

      has_list

      END EOL {
         $$ = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>($$);

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
;


class_def_inner:
      class_param_list
      from_clause
      EOL
  | error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
;

class_param_list:
   /* nothing */
   | OPENPAR opt_eol param_list opt_eol CLOSEPAR
   | OPENPAR opt_eol param_list error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol CLOSEPAR
      {
         COMPILER->raiseError(Falcon::e_syn_class, "", COMPILER->tempLine() );
      }
;

from_clause:
   /* nothing */
   | FROM inherit_list
;

inherit_list:
   inherit_token
   | inherit_list COMMA inherit_token
;

inherit_token:
   SYMBOL inherit_call
      {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol($1);
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            if ( $2 != 0 )
            {
               // save the carried
               Falcon::ListElement *iter = $2->begin();
               while( iter != 0 )
               {
                  Falcon::Value *val = (Falcon::Value *) iter->data();
                  idef->addParameter( val->genVarDefSym() );
                  iter = iter->next();
               }

               // dispose of the carrier
               delete $2;
            }
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
         }
      }
;

inherit_call:
   /* nothing */
      { $$ = 0; }
   |
   OPENPAR
      opt_eol inherit_param_list opt_eol /* in the constructor symbol table */
   CLOSEPAR
   {
      $$ = $3;
   }
;

inherit_param_list:
   inherit_param_token { $$ = new Falcon::ArrayDecl(); $$->pushBack( $1 ); }
   | inherit_param_list COMMA inherit_param_token { $1->pushBack( $3 ); $$ = $1; }
;

inherit_param_token:
   const_atom
   | SYMBOL
      {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( $1 );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( $1 );
         }
         $$ = new Falcon::Value( sym );
      }
   | SELF { $$ = new Falcon::Value(); $$->setSelf(); }
;

class_statement_list:
   /* nothing */
   | class_statement_list class_statement
;

class_statement:
   EOL
   | func_statement {
      COMPILER->addFunction( $1 );
   }
   | property_decl {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( $1 != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( $1 );  // this goes directly in the auto constructor.
      }
   }
   | init_decl
;

init_decl:
   INIT EOL {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }

      static_block

      statement_list

      END EOL {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      }
;

property_decl:
   STATIC SYMBOL OP_ASSIGN expression EOL
   {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = $4->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *$2;
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *$2 ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *$2 );
         else
            clsdef->addProperty( $2, new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete $4; // the expression is not needed anymore
      $$ = 0; // we don't add any statement
   }

   | SYMBOL OP_ASSIGN expression EOL
   {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = $3->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *$1 ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *$1 );
         else
            clsdef->addProperty( $1, def );
         delete $3; // the expression is not needed anymore
         $$ = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *$1 ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *$1 );
         else
            clsdef->addProperty( $1, new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         $$ = new Falcon::StmtVarDef( LINE, $1, $3 );
      }
   }
;

has_list:
   /* nothing */
   | HAS has_clause_list EOL
   | HAS error EOL
   {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   }
;

has_clause_list:
   SYMBOL
      {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( $1 ) );
      }
   | NOT SYMBOL
      {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( $2 ) );
      }
   | has_clause_list COMMA SYMBOL
      {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( $3 ) );
      }
   | has_clause_list COMMA NOT SYMBOL
      {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( $4 ) );
      }
;

/*****************************************************
   Object declaration
******************************************************/

object_decl:
   OBJECT SYMBOL
      {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *$2;
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( $2 );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( $2 );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
      object_decl_inner

      object_statement_list

      has_list

      END EOL {
         $$ = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>($$);

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
;

object_decl_inner:
   from_clause EOL
   | error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
;

object_statement_list:
   /* nothing */
   | object_statement_list object_statement
;

object_statement:
   EOL
   | func_statement {
      COMPILER->addFunction( $1 );
   }
   | property_decl {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( $1 != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( $1 );  // this goes directly in the auto constructor.
      }
   }
   | init_decl
;

/*****************************************************
   global statement
******************************************************/

global_statement:
   GLOBAL
      {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
      global_symbol_list EOL
      {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         $$ = COMPILER->getContext();
         COMPILER->popContext();
      }
;

global_symbol_list:
   globalized_symbol
   | global_symbol_list COMMA globalized_symbol
   | global_symbol_list COMMA error
      {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
;

globalized_symbol:
   SYMBOL
      {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( $1 );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
;

/*****************************************************
   return statement
******************************************************/

return_statement:
   RETURN EOL { $$ = new Falcon::StmtReturn(LINE, 0); }
   | RETURN expression EOL { $$ = new Falcon::StmtReturn( LINE, $2 ); }
   | RETURN error EOL { COMPILER->raiseError(Falcon::e_syn_return ); $$ = 0; }
;

/*****************************************************
   Grammar tokens
******************************************************/



const_atom:
   NIL { $$ = new Falcon::Value(); }
   | INTNUM { $$ = new Falcon::Value( $1 ); }
   | DBLNUM { $$ = new Falcon::Value( $1 ); }
   | STRING { $$ = new Falcon::Value( $1 ); }
;

atomic_symbol:
   SYMBOL
      {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( $1 );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( $1 );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         $$ = val;
     }
;

var_atom:
   atomic_symbol
   | SELF { $$ = new Falcon::Value(); $$->setSelf(); }
   | SENDER { $$ = new Falcon::Value(); $$->setSender(); }
;

/* Currently not needed
atom:
   const_atom
   | var_atom
;
*/

variable:
   var_atom
   | variable range_decl{
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, $1, $2 );
         $$ = new Falcon::Value( exp );
      }

   | variable OPENSQUARE expression CLOSESQUARE {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, $1, $3 );
         $$ = new Falcon::Value( exp );
      }

   | variable OPENSQUARE STAR expression CLOSESQUARE {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, $1, $4 );
         $$ = new Falcon::Value( exp );
      }


   | variable DOT SYMBOL {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, $1, new Falcon::Value( $3 ) );
         $$ = new Falcon::Value( exp );
      }

;


expression:
     const_atom
   | variable
   | expression PLUS expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, $1, $3 ) ); }
   | MINUS expression %prec NEG { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, $2 ) ); }
   | expression MINUS expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, $1, $3 ) ); }
   | expression STAR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, $1, $3 ) ); }
   | expression SLASH expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, $1, $3 ) ); }
   | expression PERCENT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, $1, $3 ) ); }
   | expression POW expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, $1, $3 ) ); }
   | expression AMPER expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, $1, $3 ) ); }
   | expression VBAR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, $1, $3 ) ); }
   | expression CAP expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, $1, $3 ) ); }
   | expression SHL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, $1, $3 ) ); }
   | expression SHR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, $1, $3 ) ); }
   | BANG expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, $2 ) ); }
   | LET variable OP_EQ expression { COMPILER->defineVal( $2 ); $$ =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, $2, $4 ) ); }
   | LET variable OP_ASSIGN expression { COMPILER->defineVal( $2 ); $$ =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, $2, $4 ) ); }
   | expression NEQ expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, $1, $3 ) ); }
   | expression INCREMENT { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, $1 ) ); }
   | INCREMENT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, $2 ) ); }
   | expression DECREMENT { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, $1 ) ); }
   | DECREMENT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, $2 ) ); }
   | expression EEQ expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, $1, $3 ) ); }
   | expression OP_EQ expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, $1, $3 ) ); }
   | expression OP_ASSIGN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, $1, $3 ) ); }
   | expression GT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, $1, $3 ) ); }
   | expression LT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, $1, $3 ) ); }
   | expression GE expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, $1, $3 ) ); }
   | expression LE expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, $1, $3 ) ); }
   | expression AND expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, $1, $3 ) ); }
   | expression OR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, $1, $3 ) ); }
   | NOT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, $2 ) ); }
   | expression HAS expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, $1, $3 ) ); }
   | expression HASNT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, $1, $3 ) ); }
   | expression OP_IN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, $1, $3 ) ); }
   | expression OP_NOTIN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, $1, $3 ) ); }
   | expression PROVIDES SYMBOL { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, $1, new Falcon::Value( $3 ) ) ); }
   | DOLLAR expression { $$ = new Falcon::Value( $2 ); }
   | ATSIGN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, $2 ) ); }
   | DIESIS expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, $2 ) ); }
   | lambda_expr
   | func_call
   | func_call DOT SYMBOL {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, $1, new Falcon::Value( $3 ) );
         $$ = new Falcon::Value( exp );
      }
   | func_call OPENSQUARE expression CLOSESQUARE {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, $1, $3 );
         $$ = new Falcon::Value( exp );
      }
   | func_call OPENSQUARE STAR expression CLOSESQUARE {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, $1, $4 );
         $$ = new Falcon::Value( exp );
      }
   | func_call range_decl{
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, $1, $2 );
         $$ = new Falcon::Value( exp );
      }
   | iif_expr
   | array_decl /*suqared expr*/
   | dict_decl /*suqared expr*/
   | range_decl /*suqared expr*/
   | OPENPAR opt_eol expression opt_eol CLOSEPAR { $$ = $3; }
;

/*suqared expr NEED to start with an opt_eol or with a nonambiguous symbol */
range_decl:
   OPENSQUARE COLON CLOSESQUARE {
         $$ = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
   | OPENSQUARE expression COLON CLOSESQUARE {
         $$ = new Falcon::Value( new Falcon::RangeDecl( $2 ) );
      }
   | OPENSQUARE COLON expression CLOSESQUARE {
         $$ = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), $3 ) );
      }
   | OPENSQUARE expression COLON expression CLOSESQUARE {
         $$ = new Falcon::Value( new Falcon::RangeDecl( $2, $4 ) );
      }
;

func_call:
   expression OPENPAR opt_eol par_expression_list opt_eol CLOSEPAR
      {
         $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      $1, new Falcon::Value( $4 ) ) );
      }

   | expression OPENPAR opt_eol CLOSEPAR
      {
         $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, $1, 0 ) );
      }

   | expression OPENPAR opt_eol par_expression_list error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol  CLOSEPAR
      {
         delete $4;
         COMPILER->raiseError(Falcon::e_syn_funcall, "", COMPILER->tempLine() );
         $$ = new Falcon::Value;
      }
;

opt_eol:
   /* Nothing */
   | eol_seq
;

eol_seq:
   EOL
   | eol_seq EOL
;

lambda_expr:
   LAMBDA
      {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
      lambda_decl_inner
      static_block

      statement_list

      END {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         }
;

lambda_decl_inner:
   OPENPAR opt_eol param_list opt_eol CLOSEPAR EOL
   | OPENPAR opt_eol param_list error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol CLOSEPAR EOL
      {
         COMPILER->raiseError(Falcon::e_syn_lambda );
      }
   | error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_lambda );
      }
;

iif_expr:
   expression QUESTION expression COLON expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_iif, $1, $3, $5 ) ); }
   | expression QUESTION error
      {
         delete $1;
         COMPILER->raiseError(Falcon::e_syn_iif );
         $$ = new Falcon::Value;
      }
;


array_decl:
   OPENSQUARE CLOSESQUARE {  $$ = new Falcon::Value( new Falcon::ArrayDecl() ); }
   | OPENSQUARE par_expression_list opt_eol CLOSESQUARE
      {
         $$ = new Falcon::Value( $2 );
      }
   | OPENSQUARE par_expression_list error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol CLOSESQUARE
      {
         COMPILER->raiseError(Falcon::e_syn_arraydecl, "", COMPILER->tempLine() );
         $$ = new Falcon::Value( $2 );
      }
;

dict_decl:
   OPENSQUARE ARROW CLOSESQUARE {  $$ = new Falcon::Value( new Falcon::DictDecl() ); }
   | OPENSQUARE expression_pair_list opt_eol CLOSESQUARE { $$ = new Falcon::Value( $2 ); }
   | OPENSQUARE expression_pair_list error { COMPILER->tempLine( CURRENT_LINE ); } opt_eol CLOSESQUARE
      {
         COMPILER->raiseError(Falcon::e_syn_dictdecl, "", COMPILER->tempLine() );
         $$ = new Falcon::Value( $2 );
      }
;

expression_list:
   expression { $$ = new Falcon::ArrayDecl(); $$->pushBack( $1 ); }
   | expression_list COMMA expression { $1->pushBack( $3 ); $$ = $1; }
;

par_expression_list:
   expression { $$ = new Falcon::ArrayDecl(); $$->pushBack( $1 ); }
   | par_expression_list COMMA opt_eol expression { $1->pushBack( $4 ); $$ = $1; }
;

symbol_list:
   atomic_symbol {
         COMPILER->defineVal( $1 );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( $1 );
         COMPILER->defineVal( $1 );
         $$ = ad;
      }
   | symbol_list COMMA atomic_symbol {
         COMPILER->defineVal( $3 );
         $1->pushBack( $3 );
      }
;

expression_pair_list:
   expression ARROW expression { $$ = new Falcon::DictDecl(); $$->pushBack( $1, $3 ); }
   | expression_pair_list COMMA opt_eol expression ARROW expression { $1->pushBack( $4, $6 ); $$ = $1; }
;


%% /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */

