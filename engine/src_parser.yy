/*
   FALCON - The Falcon Programming Language.
   FILE: src_parser.yy

   Bison grammar definition for falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven mag 21 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
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
   Falcon::List *fal_genericList;
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
%token UNB
%token END
%token DEF
%token WHILE BREAK CONTINUE DROPPING
%token IF ELSE ELIF
%token FOR
%token FORFIRST FORLAST FORMIDDLE
%token SWITCH CASE DEFAULT
%token SELECT
%token SELF FSELF
%token TRY CATCH RAISE
%token CLASS FROM OBJECT
%token RETURN
%token GLOBAL
%token INIT
%token LOAD
%token LAUNCH
%token CONST_KW
%token EXPORT
%token IMPORT
%token DIRECTIVE
%token COLON
%token FUNCDECL STATIC
%token INNERFUNC
%token FORDOT
%token LISTPAR
%token LOOP
%token ENUM
%token TRUE_TOKEN
%token FALSE_TOKEN

/* Special token used by the parser to generate a parse print */
%token <stringp> OUTER_STRING

%token CLOSEPAR OPENPAR CLOSESQUARE OPENSQUARE DOT OPEN_GRAPH CLOSE_GRAPH
/*
   Assigning rule precendence: immediate operations have maximum precedence, being resolved immediately,
   then the assignment gets more precendece than the expressions (OP_EQ is preferibily parsed as an
   assignment request where ambiguity arises).
*/

%left ARROW
%right VBAR
%right ASSIGN_ADD ASSIGN_SUB ASSIGN_MUL ASSIGN_DIV ASSIGN_MOD ASSIGN_BAND ASSIGN_BOR ASSIGN_BXOR ASSIGN_SHR ASSIGN_SHL ASSIGN_POW
%right OP_EQ
%right COMMA OP_TO OP_AS
%left QUESTION
%right COLON
%left OR
%left AND
%right NOT
%left OP_EXEQ EEQ NEQ GT LT GE LE
%left OP_IN OP_NOTIN PROVIDES
%right ATSIGN DIESIS
%left VBAR_VBAR CAP_CAP
%left AMPER_AMPER
%left PLUS MINUS
%left STAR SLASH PERCENT
%left POW
%left SHL SHR
%right NEG TILDE CAP_EVAL CAP_OOB CAP_DEOOB CAP_ISOOB CAP_XOROOB
%right DOLLAR INCREMENT DECREMENT AMPER

%type <integer> INTNUM_WITH_MINUS
%type <fal_adecl> expression_list listpar_expression_list array_decl
%type <fal_adecl> symbol_list
%type <fal_ddecl> expression_pair_list
%type <fal_genericList> import_symbol_list
%type <fal_val> expression func_call nameless_func innerfunc nameless_block iif_expr
%type <fal_val> for_to_expr for_to_step_clause loop_terminator
%type <fal_val> switch_decl select_decl while_decl while_short_decl
%type <fal_val> if_decl if_short_decl elif_decl
%type <fal_val> const_atom const_atom_non_minus var_atom  atomic_symbol /* atom */
%type <fal_val> dotarray_decl dict_decl
%type <fal_val> inherit_call
%type <fal_stat> break_statement continue_statement
%type <fal_stat> toplevel_statement statement while_statement if_statement
%type <fal_stat> forin_statement switch_statement fordot_statement forin_header
%type <fal_stat> select_statement
%type <fal_stat> try_statement raise_statement return_statement global_statement
%type <fal_stat> base_statement
%type <fal_stat> launch_statement
%type <fal_stat> func_statement
%type <fal_stat> self_print_statement
%type <fal_stat> enum_statement
%type <fal_stat> class_decl object_decl property_decl state_decl export_statement directive_statement
%type <fal_stat> import_statement
%type <fal_stat> def_statement
%type <fal_stat> loop_statement
%type <fal_stat> outer_print_statement


%type <fal_stat> const_statement
%type <fal_val>  range_decl

%%

/****************************************************
* Rules for falcon.
*****************************************************/

/*
input:
   {yydebug = 1; } body
;*/


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
   load_statement { $$=0; }
   | directive_statement
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
   | enum_statement
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
   | import_statement /* no action */
;

INTNUM_WITH_MINUS:
   INTNUM
   | MINUS INTNUM %prec NEG { $$ = - $2; }
;

load_statement:
   LOAD SYMBOL EOL
      {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *$2, false );
      }
   | LOAD STRING EOL
      {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *$2, true );
      }
   | LOAD error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
;

statement:
   base_statement { COMPILER->checkLocalUndefined(); $$ = $1; }
   | attribute_statement {$$=0;}/* no action */
   | EOL { $$ = 0; }
   | FUNCDECL error EOL { COMPILER->raiseError(Falcon::e_toplevel_func ); $$ = 0; }
   | OBJECT error EOL { COMPILER->raiseError(Falcon::e_toplevel_obj ); $$ = 0; }
   | CLASS error EOL { COMPILER->raiseError(Falcon::e_toplevel_class ); $$ = 0; }
   | error EOL { COMPILER->raiseError(Falcon::e_syntax ); $$ = 0;}
;


base_statement:
   expression EOL {
         if( (! COMPILER->isInteractive()) && ((!$1->isExpr()) || (!$1->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         $$ = new Falcon::StmtAutoexpr( LINE, $1 );
      }

   | expression_list OP_EQ expression EOL {
         Falcon::Value *first = new Falcon::Value( $1 );
         COMPILER->defineVal( first );
         $$ = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, $3 ) ) );
      }
   | expression_list OP_EQ expression COMMA expression_list EOL {
         if ( $1->size() != $5->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( $1 );

         COMPILER->defineVal( first );
         $5->pushFront( $3 );
         Falcon::Value *second = new Falcon::Value( $5 );
         $$ = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      }

   | def_statement /* no action  -- at the moment def_statement always returns 0*/
   | while_statement
   | loop_statement
   | forin_statement
   | switch_statement
   | select_statement
   | if_statement
   | break_statement
   | continue_statement
   | try_statement
   | raise_statement
   | return_statement
   | global_statement
   | launch_statement
   | fordot_statement
   | self_print_statement
   | outer_print_statement
;

assignment_def_list:
   assignment_def_list_element
   | assignment_def_list COMMA assignment_def_list_element 
;

assignment_def_list_element:
   atomic_symbol 
      {
      COMPILER->defContext( true );
      COMPILER->defineVal( $1 );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, $1, new Falcon::Value() ) ) ) );
      } 
   
   | atomic_symbol OP_EQ expression
      {
      COMPILER->defContext( true );
      COMPILER->defineVal( $1 );      
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, $1, $3 ) ) ) );
      }
;

def_statement:
   DEF assignment_def_list EOL
      { COMPILER->defContext( false );  $$=0; }
   | DEF error EOL
      { COMPILER->raiseError( Falcon::e_syn_def ); }
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
   | WHILE error EOL { COMPILER->raiseError(Falcon::e_syn_while ); $$ = 0; }
;

while_short_decl:
   WHILE expression COLON { $$ = $2; }
   | WHILE error COLON { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); $$ = 0; }
;

loop_statement:
   LOOP EOL {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
      statement_list END loop_terminator EOL
      {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition($6);
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = w;
      }
   | LOOP COLON statement {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( $3 != 0 )
            w->children().push_back( $3 );
         $$ = w;
      }
   | LOOP error EOL {
      COMPILER->raiseError( Falcon::e_syn_loop );
      $$ = 0;
   }
;

loop_terminator:
   { $$=0; }
   | expression { $$ = $1; }
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

forin_header:
   FOR symbol_list OP_IN expression
      {
         $$ = new Falcon::StmtForin( LINE, $2, $4 );
      }
      
   | FOR atomic_symbol OP_EQ for_to_expr
      {
         COMPILER->defineVal( $2 );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( $2 );
         $$ = new Falcon::StmtForin( LINE, decl, $4 );  
      }
   
   | FOR symbol_list OP_IN error EOL
      { delete $2;
         COMPILER->raiseError( Falcon::e_syn_forin );
         $$ = 0;
      }
   | FOR error EOL
      {
         COMPILER->raiseError( Falcon::e_syn_forin );
         $$ = 0;
      }
;
   
forin_statement:
   
   forin_header COLON 
   {
      Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>($1);
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
    }
    statement
    {
         if( $4 != 0 )
         {
            COMPILER->addStatement( $4 );
         }
         
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         $$ = $1;
   }

   | forin_header EOL
      {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>($1);
      
      
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
;

for_to_expr:
   expression OP_TO expression for_to_step_clause
      {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( $1,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, $3)), $4 );
         $$ = new Falcon::Value( rd );
      }
   | expression OP_TO expression error
      {
         $$ = new Falcon::Value( new Falcon::RangeDecl( $1, $3, 0 ) );
      }
   | expression OP_TO  error
      {
         $$ = new Falcon::Value( new Falcon::RangeDecl( $1, 0, 0 ) );
      }
;

for_to_step_clause:
   /* nothing */ { $$=0; }
   | COMMA expression { $$=new Falcon::Value( $2 ); }
   | COMMA error  { $$=0; }
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
   | middle_loop_block
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
   SHR  expression_list EOL
      {
         $$ = new Falcon::StmtSelfPrint( LINE, $2 );
      }
   | SHR EOL
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

   | SHR error EOL
      {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         $$ = 0;
      }
   | GT error EOL
      {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         $$ = 0;
      }
;


outer_print_statement:
   OUTER_STRING
   {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( $1 ) );
      $$ = new Falcon::StmtSelfPrint( LINE, adecl );
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
         if ( $3 != 0 )
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
         if ( $3 != 0 )
            f->lastBlock().push_back( $3 );
      }
   | FORLAST error EOL { COMPILER->raiseError(Falcon::e_syn_forlast ); }
;

middle_loop_block:
   FORMIDDLE EOL {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      }
      statement_list
      END EOL
      { COMPILER->popContextSet(); }

   | FORMIDDLE COLON statement {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( $3 != 0 )
            f->middleBlock().push_back( $3 );
      }
   | FORMIDDLE error EOL { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
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

   | INTNUM_WITH_MINUS
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

   | INTNUM_WITH_MINUS OP_TO INTNUM_WITH_MINUS
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
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *$1 );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( *$1 );
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
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *$1 );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( *$1 );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
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
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *$1 );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *$1 );
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
   func_begin OPENPAR param_list CLOSEPAR EOL
   | func_begin OPENPAR param_list error { COMPILER->tempLine( CURRENT_LINE ); } CLOSEPAR EOL
      {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
   | func_begin error EOL { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
;

func_decl_short:
   func_begin OPENPAR param_list CLOSEPAR COLON
   | func_begin OPENPAR error { COMPILER->tempLine( CURRENT_LINE ); } CLOSEPAR COLON
      {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
;

func_begin:
   FUNCDECL SYMBOL
      {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            func_name = stmt_cls->symbol()->name() + "." + *$2;
         }
         else if ( parent != 0 && parent->type() == Falcon::Statement::t_state ) 
         {
            Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );
            func_name =  
                  stmt_state->owner()->symbol()->name() + "." + 
                  * stmt_state->name() + "#" + *$2;
         }
         else
            func_name = *$2;

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
         if ( parent != 0 )
         {
            if( parent->type() == Falcon::Statement::t_class ) 
            {
               Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
               Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
               if ( cd->hasProperty( *$2 ) ) {
                  COMPILER->raiseError(Falcon::e_prop_adef, *$2 );
               }
               else 
               {
                   cd->addProperty( $2, new Falcon::VarDef( sym ) );
                   // is this a setter/getter?
                   if( ( $2->find( "__set_" ) == 0 || $2->find( "__get_" ) == 0 ) && $2->length() > 6 )
                   {
                      Falcon::String *pname = COMPILER->addString( $2->subString( 6 ));
                      Falcon::VarDef *pd = cd->getProperty( *pname );
                      if( pd == 0 )
                      {
                        pd = new Falcon::VarDef;
                        cd->addProperty( pname, pd );
                        pd->setReflective( Falcon::e_reflectSetGet, 0xFFFFFFFF );
                      }
                      else if( ! pd->isReflective() )
                      {
                        COMPILER->raiseError(Falcon::e_prop_adef, *pname );
                      }
   
                   }
               }
            }
            else if ( parent->type() == Falcon::Statement::t_state ) 
            {
   
               Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );            
               if( ! stmt_state->addFunction( $2, sym ) )
               {
                  COMPILER->raiseError(Falcon::e_sm_adef, *$2 );
               }
               else 
               {
                  stmt_state->state()->addFunction( *$2, sym );
               }
               
               // eventually add a property where to store this thing
               Falcon::ClassDef *cd = stmt_state->owner()->symbol()->getClassDef();
               if ( ! cd->hasProperty( *$2 ) )
                  cd->addProperty( $2, new Falcon::VarDef );
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
   | param_list COMMA param_symbol
;

param_symbol:
   SYMBOL
      {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *$1 );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), *$1 );
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
   Const Statement
***********************************************************/

const_statement:
   CONST_KW SYMBOL OP_EQ const_atom EOL
      {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = $4; //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *$2, val, LINE );
         // we don't need the expression anymore
         // no other action:
         $$ = 0;
      }
   | CONST_KW SYMBOL OP_EQ error EOL
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
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( *$1 );
         sym->exported(true);
      }
   | export_symbol_list COMMA SYMBOL
      {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( *$3 );
         sym->exported(true);
      }
;

import_statement:
   IMPORT import_symbol_list EOL
      {
         COMPILER->importSymbols( $2 );
         $$ = 0;
      }
   | IMPORT import_symbol_list FROM SYMBOL EOL
      {
         COMPILER->importSymbols( $2, *$4, "", false );
         $$ = 0;
      }
   | IMPORT import_symbol_list FROM STRING EOL
      {
         COMPILER->importSymbols( $2, *$4, "", true );
         $$ = 0;
      }
   | IMPORT import_symbol_list FROM SYMBOL OP_AS SYMBOL EOL
      {
         // destroy the list to avoid leak
         Falcon::ListElement *li = $2->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( *symName, *$4, *$6, false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete $2;

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         $$ = 0;
      }
   | IMPORT import_symbol_list FROM STRING OP_AS SYMBOL EOL
      {
         // destroy the list to avoid leak
         Falcon::ListElement *li = $2->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( *symName, *$4, *$6, true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete $2;

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         $$ = 0;
      }
   | IMPORT import_symbol_list FROM SYMBOL OP_IN SYMBOL EOL
      {
         COMPILER->importSymbols( $2, *$4, *$6, false );
         $$ = 0;
      }
   | IMPORT import_symbol_list FROM STRING OP_IN SYMBOL EOL
      {
         COMPILER->importSymbols( $2, *$4, *$6, true );
         $$ = 0;
      }
   | IMPORT SYMBOL error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_import );
         $$ = 0;
      }
   | IMPORT import_symbol_list error EOL
      {
         // destroy the list to avoid leak
         Falcon::ListElement *li = $2->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete $2;

         COMPILER->raiseError(Falcon::e_syn_import );
         $$ = 0;
      }
   | IMPORT FROM SYMBOL EOL
      {
         COMPILER->addNamespace( *$3, "", true, false );
         $$ = 0;
      }
   | IMPORT FROM STRING EOL
      {
         COMPILER->addNamespace( *$3, "", true, true );
         $$ = 0;
      }
   | IMPORT FROM SYMBOL OP_AS SYMBOL EOL
      {
         COMPILER->addNamespace( *$3, *$5, true, false );
         $$ = 0;
      }
   | IMPORT FROM STRING OP_AS SYMBOL EOL
      {
         COMPILER->addNamespace( *$3, *$5, true, true );
         $$ = 0;
      }
   | IMPORT error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_import );
         $$ = 0;
      }
;


attribute_statement:
   SYMBOL COLON const_atom EOL
     {
      COMPILER->addAttribute( *$1, $3, LINE );
     }

   | SYMBOL COLON error EOL
     {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
;

import_symbol_list:
   SYMBOL
      {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *$1 ) );
         $$ = lst;
      }
   | import_symbol_list COMMA SYMBOL
      {
         $1->pushBack( new Falcon::String( *$3 ) );
         $$ = $1;
      }
;

/**********************************************************
   Directive directive (no, it's not an error)
***********************************************************/

directive_statement:
   DIRECTIVE directive_pair_list EOL
      {
         // no effect
         $$=0;
      }
   | DIRECTIVE error EOL
     {
         COMPILER->raiseError(Falcon::e_syn_directive );
         $$=0;
     }
;

directive_pair_list:
   directive_pair
   | directive_pair_list COMMA directive_pair
;

directive_pair:
   SYMBOL OP_EQ SYMBOL
      {
         COMPILER->setDirective( *$1, *$3 );
      }
   | SYMBOL OP_EQ STRING
      {
         COMPILER->setDirective( *$1, *$3 );
      }
   | SYMBOL OP_EQ INTNUM_WITH_MINUS
      {
         COMPILER->setDirective( *$1, $3 );
      }
;



/**********************************************************
   Class Declaration
***********************************************************/

class_decl:
      CLASS SYMBOL
      {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *$2 );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *$2 );
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

      END EOL {
         $$ = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>($$);

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
            {
               Falcon::ClassDef *cd = cls->symbol()->getClassDef();
               if ( ! cd->inheritance().empty() )
               {
                  COMPILER->buildCtorFor( cls );
                  // COMPILER->addStatement( func ); should be done in buildCtorFor
                  // cls->ctorFunction( func ); idem
               }
            }
            COMPILER->popContext();
            //We didn't pushed a context set
            COMPILER->popFunction();
         }
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
   | OPENPAR param_list CLOSEPAR
   | OPENPAR param_list error { COMPILER->tempLine( CURRENT_LINE ); } CLOSEPAR
      {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
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
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol(*$1);
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), $2 ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete $2;
         }
      }
;

inherit_call:
   /* nothing */
      { $$ = 0; }
   | OPENPAR CLOSEPAR { $$=0; }
   | OPENPAR expression_list  CLOSEPAR
   {
      $$ = $2 == 0 ? 0 : new Falcon::Value( $2 );
   }
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
   
   | state_decl
      {
         Falcon::StmtState* ss = static_cast<Falcon::StmtState *>($1);
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );         
         if( ! cls->addState( static_cast<Falcon::StmtState *>($1) ) )
         {
            const Falcon::String* name = ss->name();
            // we're not using the state we created, afater all.
            delete ss->state();
            delete $1;
            COMPILER->raiseError( Falcon::e_state_adef, *name ); 
         }
         else {
            // ownership passes on to the classdef
            cls->symbol()->getClassDef()->addState( ss->name(), ss->state() );
            cls->symbol()->getClassDef()->addProperty( ss->name(), 
               new Falcon::VarDef( ss->name() ) );
         }
      }
   
   | init_decl
   | attribute_statement
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
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }

      static_block

      statement_list

      END EOL {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
;

property_decl:
   STATIC SYMBOL OP_EQ const_atom EOL
   {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = $4->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *$2;
         Falcon::Symbol *sym = COMPILER->addGlobalVar( prop_name, def );
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

   | SYMBOL OP_EQ expression EOL
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

state_decl:
   state_heading
   
   state_statement_list
   
   END EOL 
      { 
         $$ = COMPILER->getContext(); 
         COMPILER->popContext();
      }
;

state_heading:
   OPENSQUARE SYMBOL CLOSESQUARE EOL
      {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
            
         COMPILER->pushContext( 
            new Falcon::StmtState( $2, cls ) ); 
      }
  | OPENSQUARE INIT CLOSESQUARE EOL
      {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
         
         Falcon::StmtState* state = new Falcon::StmtState( COMPILER->addString( "init" ), cls );
         cls->initState( state );
         
         COMPILER->pushContext( state ); 
      }
;


state_statement_list:
   /* nothing */
   | state_statement_list state_statement
;


state_statement:
   EOL
   | func_statement
      {
         COMPILER->addFunction( $1 );
      }
;

/*****************************************************
   ENUM declaration
******************************************************/

enum_statement:
      ENUM SYMBOL
      {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *$2 );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *$2 );
            sym->setEnum( true );
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

         COMPILER->resetEnum();
      }
      EOL

      enum_statement_list

      END EOL {
         $$ = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
;

enum_statement_list:
   /* nothing */
   | enum_statement_list enum_item_decl
;

enum_item_decl:
   EOL
   | SYMBOL OP_EQ const_atom enum_item_terminator
      {
         COMPILER->addEnumerator( *$1, $3 );
      }
   | attribute_statement
   | SYMBOL enum_item_terminator
      {
         COMPILER->addEnumerator( *$1 );
      }
;

enum_item_terminator:
   EOL | COMMA
;

/*****************************************************
   Object declaration
******************************************************/

object_decl:
   OBJECT SYMBOL
      {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *$2;
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( cl_name );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *$2 );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *$2 );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
      object_decl_inner

      object_statement_list

      END EOL {
         $$ = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>($$);

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
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
   | attribute_statement
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
   | error
      {
         COMPILER->raiseError( Falcon::e_syn_global );
      }

   | globalized_symbol error
      {
         COMPILER->raiseError( Falcon::e_syn_global );
      }

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
         Falcon::Symbol *sym = COMPILER->globalize( *$1 );

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

const_atom_non_minus:
   NIL { $$ = new Falcon::Value(); }
   | UNB { $$ = new Falcon::Value(); $$->setUnbound(); }
   | TRUE_TOKEN { $$ = new Falcon::Value( true ); }
   | FALSE_TOKEN { $$ = new Falcon::Value( false ); }
   | INTNUM { $$ = new Falcon::Value( $1 ); }
   | DBLNUM { $$ = new Falcon::Value( $1 ); }
   | STRING { $$ = new Falcon::Value( $1 ); }
;

const_atom:
   NIL { $$ = new Falcon::Value(); }
   | UNB { $$ = new Falcon::Value(); $$->setUnbound(); }
   | TRUE_TOKEN { $$ = new Falcon::Value( true ); }
   | FALSE_TOKEN { $$ = new Falcon::Value( false ); }
   | INTNUM_WITH_MINUS { $$ = new Falcon::Value( $1 ); }
   | DBLNUM { $$ = new Falcon::Value( $1 ); }
   | STRING { $$ = new Falcon::Value( $1 ); }
;

atomic_symbol:
   SYMBOL
      {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *$1 );
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
   | FSELF {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         $$ = new Falcon::Value();
      }
      else
      {
         $$ = new Falcon::Value( sfunc->symbol() );
      }
   }
;

/* Currently not needed
atom:
   const_atom
   | var_atom
;
*/

OPT_EOL:
   /* nothing */
   |EOL
;

expression:
     const_atom_non_minus
   | var_atom
   | AMPER SYMBOL { $$ = new Falcon::Value(); $$->setLBind( $2 ); /* do not add the symbol to the compiler */ }
   | AMPER INTNUM { char space[32]; sprintf(space, "%d", (int)$2 ); $$ = new Falcon::Value(); $$->setLBind( COMPILER->addString(space) ); }
   | AMPER SELF { $$ = new Falcon::Value(); $$->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
   | AMPER DOT SYMBOL { $$ = new Falcon::Value(); $3->prepend( "." ); $$->setLBind( $3 ); /* do not add the symbol to the compiler */ }
   | AMPER DOT INTNUM { char space[32]; sprintf(space, ".%d", (int)$3 ); $$ = new Falcon::Value(); $$->setLBind( COMPILER->addString(space) ); }
   | AMPER DOT SELF { $$ = new Falcon::Value(); $$->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
   | MINUS expression %prec NEG { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, $2 ) ); }
   | SYMBOL VBAR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value($1), $3) ); }
   | expression PLUS OPT_EOL expression {
            // is this an immediate string sum ?
            if ( $1->isString() )
            {
               if ( $4->isString() )
               {
                  Falcon::String str( *$1->asString() );
                  str += *$4->asString();
                  $1->setString( COMPILER->addString( str ) );
                  delete $4;
                  $$ = $1;
               }
               else if ( $4->isInteger() )
               {
                  Falcon::String str( *$1->asString() );
                  str.writeNumber( $4->asInteger() );
                  $1->setString( COMPILER->addString( str ) );
                  delete $4;
                  $$ = $1;
               }
               else
                  $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, $1, $4 ) );
            }
            else
               $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, $1, $4 ) );
         }
   | expression MINUS OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, $1, $4 ) ); }
   | expression STAR OPT_EOL expression {
            if ( $1->isString() )
            {
               if ( $4->isInteger() )
               {
                  Falcon::String str( $1->asString()->length() );
                  for( int i = 0; i < $4->asInteger(); ++i )
                  {
                     str.append( *$1->asString()  );
                  }
                  $1->setString( COMPILER->addString( str ) );
                  delete $4;
                  $$ = $1;
               }
               else
                  $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, $1, $4 ) );
            }
            else
               $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, $1, $4 ) );
      }
   | expression SLASH OPT_EOL expression {
            if ( $1->isString() )
            {
               if( $1->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {

                  if ( $4->isInteger() )
                  {
                     Falcon::String str( *$1->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + $4->asInteger() );
                     $1->setString( COMPILER->addString( str ) );
                     delete $4;
                     $$ = $1;
                  }
                  else
                     $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, $1, $4 ) );
               }
            }
            else
               $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, $1, $4 ) );
      }
   | expression PERCENT OPT_EOL expression {
            if ( $1->isString() )
            {
               if ( $4->isInteger() )
               {
                  Falcon::String str( *$1->asString() );
                  str.append( (Falcon::uint32) $4->asInteger() );
                  $1->setString( COMPILER->addString( str ) );
                  delete $4;
                  $$ = $1;
               }
               else
                  $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, $1, $4 ) );
            }
            else
               $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, $1, $4 ) );
      }
   | expression POW OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, $1, $4 ) ); }
   | expression AMPER_AMPER OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, $1, $4 ) ); }
   | expression VBAR_VBAR OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, $1, $4 ) ); }
   | expression CAP_CAP OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, $1, $4 ) ); }
   | expression SHL OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, $1, $4 ) ); }
   | expression SHR OPT_EOL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, $1, $4 ) ); }
   | TILDE expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, $2 ) ); }
   | expression NEQ expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, $1, $3 ) ); }
   | expression INCREMENT { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, $1 ) ); }
   | INCREMENT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, $2 ) ); }
   | expression DECREMENT { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, $1 ) ); }
   | DECREMENT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, $2 ) ); }
   | expression EEQ expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, $1, $3 ) ); }
   | expression OP_EXEQ expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_exeq, $1, $3 ) ); }
   | expression GT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, $1, $3 ) ); }
   | expression LT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, $1, $3 ) ); }
   | expression GE expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, $1, $3 ) ); }
   | expression LE expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, $1, $3 ) ); }
   | expression AND expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, $1, $3 ) ); }
   | expression OR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, $1, $3 ) ); }
   | NOT expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, $2 ) ); }
   | expression OP_IN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, $1, $3 ) ); }
   | expression OP_NOTIN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, $1, $3 ) ); }
   | expression PROVIDES SYMBOL { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, $1, new Falcon::Value( $3 ) ) ); }
   | DOLLAR atomic_symbol { $$ = new Falcon::Value( $2 ); }
   | DOLLAR INTNUM { $$ = new Falcon::Value( (Falcon::Value *) 0 ); }
   | ATSIGN expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, $2 ) ); }
   | DIESIS expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, $2 ) ); }
   | CAP_EVAL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, $2 ) ); }
   | CAP_OOB expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, $2 ) ); }
   | CAP_DEOOB expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, $2 ) ); }
   | CAP_ISOOB expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, $2 ) ); }
   | CAP_XOROOB expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, $2 ) ); }
   | nameless_func
   | nameless_block
   | innerfunc
   | func_call
   | iif_expr
   | dotarray_decl

   | expression range_decl {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, $1, $2 );
         $$ = new Falcon::Value( exp );
      }

   | array_decl {
      $$ = new Falcon::Value( $1 );
   }

   | expression OPENSQUARE expression CLOSESQUARE {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, $1, $3 );
      $$ = new Falcon::Value( exp );
   }

   | expression OPENSQUARE STAR expression CLOSESQUARE {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, $1, $4 );
         $$ = new Falcon::Value( exp );
      }


   | expression DOT SYMBOL {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, $1, new Falcon::Value( $3 ) );
         if ( $3->getCharAt(0) == '_' && ! $1->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         $$ = new Falcon::Value( exp );
      }

   | dict_decl /*suqared expr*/
   | range_decl /*suqared expr*/

   | expression OP_EQ expression {
      COMPILER->defineVal( $1 );
      $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, $1, $3 ) );
   }

   | expression OP_EQ expression COMMA expression_list {
      COMPILER->defineVal( $1 );
      $5->pushFront( $3 );
      Falcon::Value *second = new Falcon::Value( $5 );
      $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, $1, second ) );
   }

   | expression ASSIGN_ADD expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, $1, $3 ) ); }
   | expression ASSIGN_SUB expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, $1, $3 ) ); }
   | expression ASSIGN_MUL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, $1, $3 ) ); }
   | expression ASSIGN_DIV expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, $1, $3 ) ); }
   | expression ASSIGN_MOD expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, $1, $3 ) ); }
   | expression ASSIGN_POW expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, $1, $3 ) ); }
   | expression ASSIGN_BAND expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, $1, $3 ) ); }
   | expression ASSIGN_BOR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, $1, $3 ) ); }
   | expression ASSIGN_BXOR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, $1, $3 ) ); }
   | expression ASSIGN_SHL expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, $1, $3 ) ); }
   | expression ASSIGN_SHR expression { $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, $1, $3 ) ); }
   | OPENPAR expression CLOSEPAR {$$=$2;}
;

/*suqared expr NEED to start with an or with a nonambiguous symbol */
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
   | OPENSQUARE expression COLON expression COLON expression CLOSESQUARE {
         $$ = new Falcon::Value( new Falcon::RangeDecl( $2, $4, $6 ) );
      }
;

func_call:
   expression OPENPAR expression_list CLOSEPAR
      {
         $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      $1, new Falcon::Value( $3 ) ) );
      }

   | expression OPENPAR CLOSEPAR
      {
         $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, $1, 0 ) );
      }

   | expression OPENPAR expression_list error { COMPILER->tempLine( CURRENT_LINE ); }  CLOSEPAR
      {
         delete $3;
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         $$ = new Falcon::Value;
      }
;

nameless_func:
   FUNCDECL
      {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );

         // Not defined?
         fassert( COMPILER->searchGlobalSymbol( name ) == 0 );
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( name );

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
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
      nameless_func_decl_inner
      static_block

      statement_list

      END {
            COMPILER->lexer()->popContext();
            $$ = COMPILER->closeClosure();
         }
;

nameless_block:
   OPEN_GRAPH
      {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );
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
      nameless_block_decl_inner
      static_block

      statement_list

      CLOSE_GRAPH {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         $$ = COMPILER->closeClosure();
      }
;

nameless_func_decl_inner:
   OPENPAR param_list CLOSEPAR EOL
   | OPENPAR param_list error
      {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
   | error EOL
      {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
;

nameless_block_decl_inner:
   param_list ARROW
   | param_list error
      {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
   | error ARROW
      {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
;

innerfunc:
   INNERFUNC
      {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );

         // Not defined?
         fassert( COMPILER->searchGlobalSymbol( name ) == 0 );
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( name );

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
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
      nameless_func_decl_inner
      static_block

      statement_list

      END {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            $$ = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
;





iif_expr:
   expression QUESTION expression COLON expression
   {
      $$ = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, $1, $3, $5 ) );
   }
   | expression QUESTION expression COLON error
   {
      delete $1;
      delete $3;
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      $$ = new Falcon::Value;
   }
   | expression QUESTION expression error
   {
      delete $1;
      delete $3;
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      $$ = new Falcon::Value;
   }
   | expression QUESTION error
      {
         delete $1;
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         $$ = new Falcon::Value;
      }
;


array_decl:
     OPENSQUARE CLOSESQUARE { $$ = new Falcon::ArrayDecl(); }
   | OPENSQUARE expression_list CLOSESQUARE
      {
         $$ = $2;
      }
   | OPENSQUARE expression_list error
      {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         $$ = $2;
      }
;

dotarray_decl:
   LISTPAR CLOSESQUARE {  $$ = new Falcon::Value( new Falcon::ArrayDecl() ); }
   | LISTPAR listpar_expression_list CLOSESQUARE
      {
         $$ = new Falcon::Value( $2 );
      }
   | LISTPAR listpar_expression_list error
      {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         $$ = new Falcon::Value( $2 );
      }
;


dict_decl:
   OPENSQUARE ARROW CLOSESQUARE {  $$ = new Falcon::Value( new Falcon::DictDecl() ); }
   | OPENSQUARE expression_pair_list CLOSESQUARE { $$ = new Falcon::Value( $2 ); }
   | OPENSQUARE expression_pair_list error CLOSESQUARE
      {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         $$ = new Falcon::Value( $2 );
      }
;

expression_list:
   expression { $$ = new Falcon::ArrayDecl(); $$->pushBack( $1 ); }
   | expression_list COMMA expression { $1->pushBack( $3 ); $$ = $1; }
;

listpar_expression_list:
   expression { $$ = new Falcon::ArrayDecl(); $$->pushBack( $1 ); }
   | listpar_expression_list listpar_comma expression { $1->pushBack( $3 ); $$ = $1; }
;

listpar_comma:
   /*nothing */ | COMMA;

symbol_list:
   atomic_symbol {
         COMPILER->defineVal( $1 );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( $1 );
         $$ = ad;
      }
   | symbol_list COMMA atomic_symbol {
         COMPILER->defineVal( $3 );
         $1->pushBack( $3 );
      }
;

expression_pair_list:
   expression ARROW expression { $$ = new Falcon::DictDecl(); $$->pushBack( $1, $3 ); }
   | expression_pair_list COMMA expression ARROW expression { $1->pushBack( $3, $5 ); $$ = $1; }
;


%% /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */

