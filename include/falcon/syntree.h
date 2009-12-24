/*
   FALCON - The Falcon Programming Language.
   FILE: syntree.h

   Syntactic tree item definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven giu 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SYNTREE_H
#define FALCON_SYNTREE_H

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/symtab.h>
#include <falcon/error.h>
#include <falcon/string.h>
#include <falcon/ltree.h>
#include <falcon/symlist.h>
#include <falcon/basealloc.h>

namespace Falcon
{

class Value;
class Expression;
class Compiler;
class Map;

/** Class storing array declarations in source code.
   This class records the content of [ v1, v2 .. vn ] declarations in
   source code; it's basically a Value * Hlist with ownership.
**/
class FALCON_DYN_CLASS ArrayDecl: public List
{
public:
   ArrayDecl();
   ArrayDecl( const ArrayDecl &other );


   ArrayDecl *clone() const { return new ArrayDecl( *this ); }
};

class FALCON_DYN_CLASS DictDecl: public List
{

public:
   typedef struct t_pair {
      Value *first;
      Value *second;
   } pair;

   DictDecl();
   DictDecl( const DictDecl &other );

   void pushBack( Value *first, Value *second );
   DictDecl *clone() const { return new DictDecl( *this ); }
};


class FALCON_DYN_CLASS RangeDecl: public BaseAlloc
{
   Value *m_rstart;
   Value *m_rend;
   Value *m_step;

public:
   RangeDecl( Value *start, Value *end = 0, Value *step = 0 ):
      m_rstart( start ),
      m_rend( end ),
      m_step( step )
   {}

   RangeDecl( const RangeDecl &other );

   ~RangeDecl();

   bool isOpen() const { return m_rend == 0; }
   Value *rangeStart() const { return m_rstart; }
   Value *rangeEnd() const { return m_rend; }
   Value *rangeStep() const { return m_step; }
   RangeDecl *clone() const { return new RangeDecl( *this ); }
};




class FALCON_DYN_CLASS Value: public BaseAlloc
{
public:
   typedef enum {
      t_nil,
      t_imm_bool,
      t_imm_integer,
      t_imm_string,
      t_imm_num,
      t_symbol,
      t_symdef,
      t_self,
      t_lbind,

      t_byref,
      t_array_decl,
      t_dict_decl,
      t_range_decl,
      t_unbound,

      t_expression
   } type_t;

private:
   type_t m_type;
   union {
      bool asBool;
      int64 asInteger;
      numeric asNumeric;
      String *asString;
      Symbol *asSymbol;
      ArrayDecl *asArray;
      DictDecl *asDict;
      RangeDecl *asRange;
      Expression *asExpr;
      Value *asRef;
   } m_content;

public:

   Value():
      m_type( t_nil )
   {}

   Value( const Value &other ) {
      copy( other );
   }

   explicit Value( bool val ):
      m_type( t_imm_bool )
   {
      m_content.asBool = val;
   }

   explicit Value( int64 val ):
      m_type( t_imm_integer )
   {
      m_content.asInteger = val;
   }

   explicit Value( numeric val ):
      m_type( t_imm_num )
   {
      m_content.asNumeric = val;
   }

   Value( String *val ):
      m_type( t_imm_string )
   {
      m_content.asString = val;
   }

   Value( Symbol *val ):
      m_type( t_symbol )
   {
      m_content.asSymbol = val;
   }

   Value( Expression *val ):
      m_type( t_expression )
   {
      m_content.asExpr = val;
   }

   Value( ArrayDecl *val ):
      m_type( t_array_decl )
   {
      m_content.asArray = val;
   }

   Value( DictDecl *val ):
      m_type( t_dict_decl )
   {
      m_content.asDict = val;
   }

   Value( RangeDecl *val ):
      m_type( t_range_decl )
   {
      m_content.asRange = val;
   }

   Value( Value *val ):
      m_type( t_byref )
   {
      m_content.asRef = val;
   }

   ~Value();

   /** Copies the value.
   */
   void copy( const Value &other );

   /** Clone constructor. */
   Value *clone() const;

   /** Transfers the value of the original to this instance.
      The original type is set to nil, so that this instance remains the sole
      owner of the deep value data.
   */
   void transfer( Value &other ) {
      m_type = other.m_type;
      m_content = other.m_content;
      other.m_type = t_nil;
   }

   type_t type() const { return m_type; }

   /** Creates a new Falcon::VarDef using the contents of this object.
      The Value() object is specific for the compiler; the VM and the module system
      does not know it. At times, it is needed to save in the module some data that
      are stored in the Falcon::Value; this is an utility function that does the job.

      If the value is not simple, i.e. it's an expression or a statement, it cannot
      be converted and the function won't create a VarDef. Also, this version
      won't create a VarDef for symbols; use genVarDefSym() if this is required.

      \return a newly allocated prodef that has the same contents a the value, or 0
         for complex values.
   */
   VarDef *genVarDef();

   bool isImmediate() const {
      return m_type == t_nil ||
             m_type == t_unbound ||
             m_type == t_imm_bool ||
             m_type == t_imm_integer ||
             m_type == t_imm_string ||
             m_type == t_imm_num;
   }

   bool isSimple() const {
      return isImmediate() ||
             m_type == t_symbol || m_type == t_symdef || m_type == t_lbind ||
             m_type == t_self;
   }

   bool isTrue() const {
      switch( m_type ) {
         case t_imm_integer: return asInteger() != 0;
         case t_imm_num:  return asNumeric() != 0.0;
         case t_imm_string: return asString()->size() != 0;
         default: return false;
      }
   }

   bool asBool() const { return m_content.asBool; }
   int64 asInteger() const { return m_content.asInteger; }
   numeric asNumeric() const { return m_content.asNumeric; }
   String *asString() const { return m_content.asString; }
   String *asSymdef() const { return m_content.asString; }
   String *asLBind() const { return m_content.asString; }
   Symbol *asSymbol() const { return m_content.asSymbol; }

   Value *asReference() const { return m_content.asRef; }
   ArrayDecl *asArray() const { return m_content.asArray; }
   DictDecl *asDict() const { return m_content.asDict; }
   RangeDecl *asRange() const { return m_content.asRange; }
   Expression *asExpr() const { return m_content.asExpr; }

   void setNil() { m_type = t_nil; }
   void setUnbound() { m_type = t_unbound; }
   void setBool( bool val ) { m_type = t_imm_bool; m_content.asBool = val; }
   void setInteger( int64 val ) { m_type = t_imm_integer; m_content.asInteger = val; }
   void setNumeric( numeric val ) { m_type = t_imm_num; m_content.asNumeric = val; }
   void setString( String *val ) { m_type = t_imm_string; m_content.asString = val; }
   void setSymdef( String *val ) { m_type = t_symdef; m_content.asString = val; }
   void setSymbol( Symbol *val ) { m_type = t_symbol; m_content.asSymbol = val; }
   void setReference( Value *val ) { m_type = t_byref; m_content.asRef = val; }
   void setExpr( Expression *val ) { m_type = t_expression; m_content.asExpr = val; }
   void setArray( ArrayDecl *val ) { m_type = t_array_decl; m_content.asArray = val; }
   void setDict( DictDecl *val ) { m_type = t_dict_decl; m_content.asDict = val; }
   void setRange( RangeDecl *val ) { m_type = t_range_decl; m_content.asRange = val; }
   void setSelf() { m_type = t_self; }
   void setLBind( String *val ) { m_type = t_lbind; m_content.asString = val; }

   bool isNil() const { return m_type == t_nil; }
   bool isUnbound() const { return m_type == t_unbound; }
   bool isBool() const { return m_type == t_imm_bool; }
   bool isInteger() const { return m_type == t_imm_integer; }
   bool isNumeric() const { return m_type == t_imm_num; }
   bool isString() const { return m_type == t_imm_string; }
   bool isSymdef() const { return m_type == t_symdef; }
   bool isSymbol() const { return m_type == t_symbol; }
   bool isReference() const { return m_type == t_byref; }
   bool isExpr() const { return m_type == t_expression; }
   bool isArray() const { return m_type == t_array_decl; }
   bool isDict() const { return m_type == t_dict_decl; }
   bool isSelf() const { return m_type == t_self; }
   bool isRange() const { return m_type == t_range_decl; }
   bool isLBind() const { return m_type == t_lbind; }

   Value &operator =( const Value &other ) {
      copy( other );
      return *this;
   }

   /** Verifies if another Falcon::Value is internally greater or equal than this one.
      Works for nil, immediates, ranges and symbols.
      The ordering between type is nil less than interger less than range
      less than string less than symbol.

      \param val the other value to be compared to this one.
      \return true if the type of the element is the same, and this item is less than val.
   */
   bool less( const Value &val ) const;
   bool operator <( const Value &val ) const { return less( val ); }


   /** Verifies if another Falcon::Value is internally equal to this one.
      Works for nil, immediates and symbols. If the contents of the parameter
      are the same as those of this obeject, returns true, else false.
      \param other the other value to be compared to this one.
      \return true if the type and content of the other element are the same as this.
   */
   bool isEqualByValue( const Value &other ) const;

   bool operator==( const Value &other ) const { return isEqualByValue( other ); }
   bool operator!=( const Value &other ) const { return !isEqualByValue( other ); }

   bool operator >=( const Value &other ) const { return ! less( other ); }
   bool operator <=( const Value &other ) const { return less( other ) || isEqualByValue( other ); }
   bool operator >( const Value &other) const { return ! ( less( other ) || isEqualByValue( other ) ); }
};



class FALCON_DYN_CLASS ValuePtrTraits: public VoidpTraits
{
public:
   virtual void copy( void *targetZone, const void *sourceZone ) const;
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

namespace traits {
   extern ValuePtrTraits &t_valueptr();
}


class FALCON_DYN_CLASS Expression: public BaseAlloc
{
public:
   typedef enum {
      t_none,
      t_neg,
      t_bin_not,
      t_not,

      t_bin_and,
      t_bin_or,
      t_bin_xor,
      t_shift_left,
      t_shift_right,
      t_and,
      t_or,

      t_plus,
      t_minus,
      t_times,
      t_divide,
      t_modulo,
      t_power,

      t_pre_inc,
      t_post_inc,
      t_pre_dec,
      t_post_dec,

      t_gt,
      t_ge,
      t_lt,
      t_le,
      t_eq,
      t_exeq,
      t_neq,

      t_has,
      t_hasnt,
      t_in,
      t_notin,
      t_provides,

      t_iif,
      t_lambda,

      t_obj_access,
      t_funcall,
      t_inherit,
      t_array_access,
      t_array_byte_access,
      t_strexpand,
      t_indirect,

      t_assign,
      t_fbind,

      t_aadd,
      t_asub,
      t_amul,
      t_adiv,
      t_amod,
      t_apow,
      t_aband,
      t_abor,
      t_abxor,
      t_ashl,
      t_ashr,

      t_eval,
      t_deoob,
      t_oob,
      t_xoroob,
      t_isoob,

      /** An optimized expression is like an unary operator */
      t_optimized
   } operator_t;

private:
   operator_t m_operator;
   Value *m_first;
   Value *m_second;
   Value *m_third;

public:
   Expression( operator_t t, Value *first, Value *second = 0, Value *third = 0 ):
      m_operator( t ),
      m_first( first ),
      m_second( second ),
      m_third( third )
   {}

   Expression( const Expression &other );
   ~Expression();

   operator_t type() const { return m_operator; }
   Value *first() const { return m_first; }
   void first( Value *f ) { delete m_first; m_first= f; }

   Value *second() const { return m_second; }
   void second( Value *s ) { delete m_second; m_second = s; }

   Value *third() const { return m_third; }
   void third( Value *t ) { delete m_third; m_third = t; }

   bool isStandAlone() const;

   bool isBinaryOperator() const
   {
      switch( m_operator )
      {
         case t_bin_and:
         case t_bin_or:
         case t_bin_xor:
         case t_shift_left:
         case t_shift_right:
         case t_and:
         case t_or:

         case t_plus:
         case t_minus:
         case t_times:
         case t_divide:
         case t_modulo:
         case t_power:

         case t_gt:
         case t_ge:
         case t_lt:
         case t_le:
         case t_eq:
         case t_neq:
         case t_exeq:

         case t_has:
         case t_hasnt:
         case t_in:
         case t_notin:
         case t_provides:
            return true;

         default:
            return false;
      }
      // returning from switch
   }
};


//=================================================================
// Statements below this line
//=================================================================

class FALCON_DYN_CLASS Statement: public SLElement
{
public:
   typedef enum {
      t_none,
      t_break,
      t_continue,
      t_launch,
      t_autoexp,
      t_return,
      t_attributes,
      t_raise,
      t_give,
      t_unref,
      t_if,
      t_elif,
      t_while,
      t_loop,
      t_forin,
      t_try,
      t_catch,
      t_switch,
      t_select,
      t_case,
      t_module,
      t_global,

      t_class,
      t_state,
      t_function,
      t_propdef,
      t_fordot,
      t_self_print

   } type_t;

protected:
   type_t m_type;
   uint32 m_line;

   Statement( type_t t ):
      m_type(t),
      m_line(0)
   {}

   Statement( int32 l, type_t t ):
      m_type(t),
      m_line(l)
   {}

public:
   Statement( const Statement &other );
   virtual ~Statement() {};

   type_t type() const { return m_type; }
   uint32 line() const { return m_line; }
   void line( uint32 l ) { m_line = l; }
   virtual Statement *clone() const =0;
};


/** Typed strong list holding statements. */
class FALCON_DYN_CLASS StatementList: public StrongList
{
public:
   StatementList() {}
   StatementList( const StatementList &other );
   ~StatementList();

   Statement *front() const { return static_cast< Statement *>( StrongList::front() ); }
   Statement *back() const { return static_cast< Statement *>( StrongList::back() ); }
   Statement *pop_front() { return static_cast< Statement *>( StrongList::pop_front() ); }
   Statement *pop_back() { return static_cast< Statement *>( StrongList::pop_back() ); }

};

class FALCON_DYN_CLASS StmtNone: public Statement
{
public:
   StmtNone( int32 l ):
      Statement( l, t_none )
   {}

   StmtNone( const StmtNone &other ):
      Statement( other )
   {}

   StmtNone *clone() const;
};


class FALCON_DYN_CLASS StmtGlobal: public Statement
{
   SymbolList m_symbols;

public:
   StmtGlobal(int line):
      Statement( line, t_global)
   {}

   StmtGlobal( const StmtGlobal &other );

   void addSymbol( Symbol *sym ) { m_symbols.pushBack( sym ); }
   SymbolList &getSymbols() { return m_symbols; }
   const SymbolList &getSymbols() const { return m_symbols; }

   virtual StmtGlobal *clone() const;
};


class FALCON_DYN_CLASS StmtUnref: public Statement
{
   Value *m_symbol;

public:
   StmtUnref( int line, Value *sym ):
      Statement( line, t_unref ),
      m_symbol( sym )
   {}

   StmtUnref( const StmtUnref &other );

   ~StmtUnref();

   Value *symbol() const { return m_symbol; }
   virtual StmtUnref *clone() const;
};

class FALCON_DYN_CLASS StmtSelfPrint: public Statement
{
  ArrayDecl *m_toPrint;

public:
   StmtSelfPrint( uint32 line, ArrayDecl *toPrint ):
      Statement( line, t_self_print ),
      m_toPrint( toPrint )
   {}

   StmtSelfPrint( const StmtSelfPrint &other );

   ~StmtSelfPrint();

   virtual StmtSelfPrint *clone() const;

   ArrayDecl *toPrint() const { return m_toPrint; }
};


class FALCON_DYN_CLASS StmtExpression: public Statement
{
   Value *m_expr;

public:
   StmtExpression( uint32 line, type_t t, Value *exp ):
      Statement( line, t ),
      m_expr( exp )
   {}

   StmtExpression( const StmtExpression &other );

   virtual ~StmtExpression();

   Value *value() const { return m_expr; }
   virtual StmtExpression *clone() const;
};

class FALCON_DYN_CLASS StmtFordot: public StmtExpression
{
public:
   StmtFordot( uint32 line, Value *exp ):
      StmtExpression( line, t_fordot, exp )
   {}

   StmtFordot( const StmtFordot &other );

   virtual StmtFordot *clone() const;
};

class FALCON_DYN_CLASS StmtAutoexpr: public StmtExpression
{
public:
   StmtAutoexpr( uint32 line, Value *exp ):
      StmtExpression( line, t_autoexp, exp )
   {}

   StmtAutoexpr( const StmtAutoexpr &other );

   virtual StmtAutoexpr *clone() const;
};


class FALCON_DYN_CLASS StmtReturn: public StmtExpression
{
public:
   StmtReturn( uint32 line, Value *exp ):
      StmtExpression( line, t_return, exp )
   {}

   StmtReturn( const StmtReturn &other ):
      StmtExpression( other )
   {}

   virtual StmtReturn *clone() const;
};


class FALCON_DYN_CLASS StmtLaunch: public StmtExpression
{
public:
   StmtLaunch( uint32 line, Value *exp ):
      StmtExpression( line, t_launch, exp )
   {}

   StmtLaunch( const StmtLaunch &other ):
      StmtExpression( other )
   {}

   virtual StmtLaunch *clone() const;
};


class FALCON_DYN_CLASS StmtRaise: public StmtExpression
{
public:
   StmtRaise( uint32 line, Value *exp ):
      StmtExpression( line, t_raise, exp )
   {}

   StmtRaise( const StmtRaise &other ):
      StmtExpression( other )
   {}

   virtual StmtRaise *clone() const;
};


class FALCON_DYN_CLASS StmtGive: public Statement
{
   ArrayDecl *m_objects;
   ArrayDecl *m_attribs;

public:
   StmtGive( uint32 line, ArrayDecl *objects, ArrayDecl *attribs ):
      Statement( line, t_give ),
      m_objects( objects ),
      m_attribs( attribs )
   {}

   StmtGive( const StmtGive &other );

   virtual ~StmtGive();

   ArrayDecl *objects() const { return m_objects; }
   ArrayDecl *attributes() const { return m_attribs; }

   virtual StmtGive *clone() const;
};


/** Loop control statements (break and continue) */
class FALCON_DYN_CLASS StmtLoopCtl: public Statement
{

public:
   StmtLoopCtl( uint32 line, type_t t ):
      Statement( line, t )
   {}

   StmtLoopCtl( const StmtLoopCtl &other );

   // pure virtual, no clone.
};

class FALCON_DYN_CLASS StmtBreak: public StmtLoopCtl
{

public:
   StmtBreak( uint32 line ):
      StmtLoopCtl( line, t_break )
   {}

   StmtBreak( const StmtBreak &other ):
      StmtLoopCtl( other )
   {}

   virtual StmtBreak *clone() const;
};

class FALCON_DYN_CLASS StmtContinue: public StmtLoopCtl
{
   bool m_dropping;

public:
   StmtContinue( uint32 line, bool dropping = false ):
      StmtLoopCtl( line, t_continue ),
      m_dropping( dropping )
   {}

   StmtContinue( const StmtContinue &other ):
      StmtLoopCtl( other )
   {}

   bool dropping() const { return m_dropping; }
   virtual StmtContinue *clone() const;
};


class FALCON_DYN_CLASS StmtBlock: public Statement
{
   StatementList m_list;

public:
   StmtBlock( uint32 line, type_t t ):
      Statement( line, t )
   {}

   StmtBlock( const StmtBlock &other );

   const StatementList &children() const { return m_list; }
   StatementList &children() { return m_list; }

   // pure virtual, no clone
};


class FALCON_DYN_CLASS StmtConditional: public StmtBlock
{
protected:
   Value *m_condition;

public:
   StmtConditional( uint32 line, type_t t, Value *cond ):
      StmtBlock( line, t ),
      m_condition( cond )
   {}

   StmtConditional( const StmtConditional &other );

   virtual ~StmtConditional();

   Value *condition() const { return m_condition; }

   // pure virtual
};

class FALCON_DYN_CLASS StmtLoop: public StmtConditional
{
public:
   StmtLoop( uint32 line, Value *cond = 0):
      StmtConditional( line, t_loop, cond )
   {}

   StmtLoop( const StmtLoop &other ):
      StmtConditional( other )
   {}

   virtual StmtLoop *clone() const;

   void setCondition( Value *cond ) { m_condition = cond; }
};


class FALCON_DYN_CLASS StmtWhile: public StmtConditional
{
public:
   StmtWhile( uint32 line, Value *cond ):
      StmtConditional( line, t_while, cond )
   {}

   StmtWhile( const StmtWhile &other ):
      StmtConditional( other )
   {}

   virtual StmtWhile *clone() const;
};


class FALCON_DYN_CLASS StmtElif: public StmtConditional
{
public:
   StmtElif( uint32 line, Value *cond ):
      StmtConditional( line, t_elif, cond )
   {}

   StmtElif( const StmtElif &other ):
      StmtConditional( other )
   {}

   virtual StmtElif *clone() const;
};


class FALCON_DYN_CLASS StmtIf: public StmtConditional
{
   StatementList m_else;
   StatementList m_elseifs;

public:
   StmtIf( uint32 line, Value *cond ):
      StmtConditional( line, t_if, cond )
   {}

   StmtIf( const StmtIf &other );

   const StatementList &elseChildren() const { return m_else; }
   StatementList &elseChildren() { return m_else; }
   const StatementList &elifChildren() const { return m_elseifs; }
   StatementList &elifChildren() { return m_elseifs; }

   virtual StmtIf *clone() const;
};


class FALCON_DYN_CLASS StmtForin: public StmtBlock
{
   StatementList m_first;
   StatementList m_last;
   StatementList m_middle;

   Value *m_source;
   ArrayDecl *m_dest;

public:
   StmtForin( uint32 line, ArrayDecl *dest, Value *source ):
      StmtBlock( line, t_forin ),
      m_source( source ),
      m_dest( dest )
   {}

   StmtForin( const StmtForin &other );

   virtual ~StmtForin();

   const StatementList &firstBlock() const { return m_first; }
   StatementList &firstBlock() { return m_first; }
   const StatementList &lastBlock() const { return m_last; }
   StatementList &lastBlock() { return m_last; }
   const StatementList &middleBlock() const { return m_middle; }
   StatementList &middleBlock() { return m_middle; }

   Value *source() const { return m_source; }
   ArrayDecl *dest() const { return m_dest; }

   virtual StmtForin *clone() const;
};


class FALCON_DYN_CLASS StmtCaseBlock: public StmtBlock
{
public:
   StmtCaseBlock( uint32 line ):
      StmtBlock( line, t_case )
   {}

   StmtCaseBlock( const StmtCaseBlock &other ):
      StmtBlock( other )
   {}

   virtual StmtCaseBlock *clone() const;
};


/** Statement switch.

*/
class FALCON_DYN_CLASS StmtSwitch: public Statement
{
   /**
      Maps of Value *, uint32
   */
   Map m_cases_int;
   Map m_cases_rng;
   Map m_cases_str;
   Map m_cases_obj;
   /** We store the objects also in a list to keep track of their declaration order */
   List m_obj_list;

   StatementList m_blocks;
   StatementList m_defaultBlock;

   int32 m_nilBlock;

   Value *m_cfr;

public:
   StmtSwitch( uint32 line, Value *expr );

   StmtSwitch( const StmtSwitch &other );

   virtual ~StmtSwitch();

   const Map &intCases() const { return m_cases_int; }
   const Map &rngCases() const { return m_cases_rng; }
   const Map &strCases() const { return m_cases_str; }
   const Map &objCases() const { return m_cases_obj; }
   const List &objList() const { return m_obj_list; }
   const StatementList &blocks() const { return m_blocks; }

   void addBlock( StmtCaseBlock *sl );

   Map &intCases() { return m_cases_int; }
   Map &rngCases() { return m_cases_rng; }
   Map &strCases() { return m_cases_str; }
   Map &objCases() { return m_cases_obj; }
   List &objList() { return m_obj_list; }
   StatementList &blocks() { return m_blocks; }

   int32 nilBlock() const { return m_nilBlock; }
   void nilBlock( int32 v ) { m_nilBlock = v; }

   const StatementList &defaultBlock() const { return m_defaultBlock; }
   StatementList &defaultBlock() { return m_defaultBlock; }

   Value *switchItem() const { return m_cfr; }

   bool addIntCase( Value *itm );
   bool addStringCase( Value *itm );
   bool addRangeCase( Value *itm );
   bool addSymbolCase( Value *itm );

   int currentBlock() const { return m_blocks.size(); }

   virtual StmtSwitch *clone() const;
};


/** Statement select.

*/
class FALCON_DYN_CLASS StmtSelect: public StmtSwitch
{
public:
   StmtSelect( uint32 line, Value *expr );
   StmtSelect( const StmtSelect &other );

   virtual StmtSelect *clone() const;
};

class FALCON_DYN_CLASS StmtCatchBlock: public StmtBlock
{
   Value *m_into;

public:
   StmtCatchBlock( uint32 line, Value *into = 0 ):
      StmtBlock( line, t_catch ),
      m_into( into )
   {}

   StmtCatchBlock( const StmtCatchBlock &other );

   ~StmtCatchBlock();

   Value *intoValue() const { return m_into; }

   virtual StmtCatchBlock *clone() const;
};


class FALCON_DYN_CLASS StmtTry: public StmtBlock
{
   Map m_cases_int;
   Map m_cases_sym;

   /** Objects must be stored in the order they are presented. */
   List m_sym_list;

   /** Handlers for non-default branches */
   StatementList m_handlers;

   /** Also catcher-into must be listed. */
   List m_into_values;

   /** Default block.
   */
   StmtCatchBlock *m_default;

public:
   StmtTry( uint32 line );
   StmtTry( const StmtTry &other );
   ~StmtTry();

   const StmtCatchBlock *defaultHandler() const { return m_default; }
   StmtCatchBlock *defaultHandler() { return m_default; }
   void defaultHandler( StmtCatchBlock *block );

   bool defaultGiven() const { return m_default != 0; }

   const StatementList &handlers() const { return m_handlers; }
   StatementList &handlers() { return m_handlers; }

   const Map &intCases() const { return m_cases_int; }
   const Map &objCases() const { return m_cases_sym; }
   const List &objList() const { return m_sym_list; }

   void addHandler( StmtCatchBlock *block );

   bool addIntCase( Value *itm );
   bool addSymbolCase( Value *itm );

   int currentBlock() const { return m_handlers.size(); }

   virtual StmtTry *clone() const;
};


class FALCON_DYN_CLASS StmtCallable: public Statement
{
   Symbol *m_name;

public:
   StmtCallable( uint32 line, type_t t, Symbol *name ):
      Statement( line, t ),
      m_name( name )
   {}

   StmtCallable( const StmtCallable &other ):
      Statement( other ),
      m_name( other.m_name )
   {}

   virtual ~StmtCallable();

   Symbol *symbol() { return m_name; }
   const Symbol *symbol() const { return m_name; }
   const String &name() const { return m_name->name(); }

   // pure virtual, no clone
};

class StmtFunction;
class StmtState;

class FALCON_DYN_CLASS StmtClass: public StmtCallable
{
   StmtFunction *m_ctor;
   bool m_initGiven;
   /** if this class is a clone it must delete it's constructor statement. */
   bool m_bDeleteCtor;

   Symbol *m_singleton;
   /** set of expressions (values, usually inherit calls) to be prepended to the constructor */
   ArrayDecl m_initExpressions;

   /** Init state */
   StmtState *m_initState;
   StatementList m_states;
public:

   StmtClass( uint32 line, Symbol *name ):
      StmtCallable( line, t_class, name ),
      m_ctor(0),
      m_initGiven( false ),
      m_bDeleteCtor( false ),
      m_singleton(0),
      m_initState(0)
   {}

   StmtClass( const StmtClass &other );
   virtual ~StmtClass();

   /** Function data that is used as a constructor.
      As properties may be initialized "randomly", we
      need a simple way to access the statements that will be generated
      in the constructor for this class.

      The function returned is a normal function that is found in
      the module function tree.
   */
   StmtFunction *ctorFunction() const { return m_ctor; }
   void ctorFunction( StmtFunction *func ) { m_ctor = func; }

   bool initGiven() const { return m_initGiven; }
   void initGiven( bool val ) { m_initGiven = val; }

   void addInitExpression( Value *expr ) { m_initExpressions.pushBack( expr ); }
   const ArrayDecl& initExpressions() const { return m_initExpressions; }
   ArrayDecl& initExpressions() { return m_initExpressions; }

   /** Singleton associated to this class, if any. */
   Symbol *singleton() const { return m_singleton; }
   void singleton( Symbol *s ) { m_singleton = s; }
   virtual StmtClass *clone() const;

   /** Return the Statement declaring the init state of this class */
   StmtState* initState() const { return m_initState; }

   /** Sets the init state of this class. */
   void initState( StmtState* m ) { m_initState = m; }

   /** Return the Statement declaring the init state of this class */
   bool addState( StmtState* m_state );

};

class FALCON_DYN_CLASS StmtFunction: public StmtCallable
{

private:
   int m_lambda_id;
   StatementList m_staticBlock;
   StatementList m_statements;
   const StmtClass *m_ctor_for;

public:

   StmtFunction( uint32 line, Symbol *name ):
      StmtCallable( line, t_function, name ),
      m_lambda_id(0),
      m_ctor_for(0)
   {}

   StmtFunction( const StmtFunction &other );

   StatementList &statements() { return m_statements; }
   const StatementList &statements() const { return m_statements; }

   StatementList &staticBlock() { return m_staticBlock; }
   const StatementList &staticBlock() const { return m_staticBlock; }

   bool hasStatic() const { return !m_staticBlock.empty(); }

   void setLambda( int id ) { m_lambda_id = id; }
   int lambdaId() const { return m_lambda_id; }
   bool isLambda() const { return m_lambda_id != 0; }

   void setConstructorFor( const StmtClass *cd ) { m_ctor_for = cd; }
   const StmtClass *constructorFor() const { return m_ctor_for; }

   virtual StmtFunction *clone() const;
};


class StmtState: public Statement
{
   const String* m_name;
   StmtClass* m_owner;
   Map m_funcs;
   StateDef* m_stateDef;

public:

   StmtState( const String* name, StmtClass* owner );
   StmtState( const StmtState& other );
   virtual ~StmtState();
   virtual StmtState* clone() const;

   /** Functions subscribed to this state, ordered by alias. */
   const Map& functions() const  { return m_funcs; }

   /** Functions subscribed to this state, ordered by alias. */
   Map& functions()  { return m_funcs; }

   /** Just a shortcut to insertion in the map.
    *    Returns false if the map exists.
    */
   bool addFunction( const String* name, Symbol* func );

   const String* name() const { return m_name; }
   StmtClass* owner() const { return m_owner; }

   StateDef* state() const { return m_stateDef; }
};

class FALCON_DYN_CLASS StmtVarDef: public Statement
{
   String *m_name;
   Value *m_value;

public:

   StmtVarDef( uint32 line, String *name, Value *value ):
      Statement( line, t_propdef ),
      m_name( name ),
      m_value( value )
   {}

   StmtVarDef( const StmtVarDef &other );
   virtual ~StmtVarDef();

   String *name() const { return m_name; }
   Value *value() const { return m_value; }

   virtual StmtVarDef *clone() const;
};

/** Source File syntactic tree.
   This tree represent a compiled program.
   Together with the module being created by the compiler during the compilation
   step, which contains the string table and the symbol table, this
   defines the internal representation of a script
 */
class FALCON_DYN_CLASS SourceTree: public BaseAlloc
{
   StatementList m_statements;
   StatementList m_functions;
   StatementList m_classes;

   bool m_exportAll;

public:

   SourceTree():
      m_exportAll( false )
   {}

   SourceTree( const SourceTree &other );

   const StatementList &statements() const { return m_statements; }
   StatementList &statements() { return m_statements; }

   const StatementList &functions() const { return m_functions; }
   StatementList &functions() { return m_functions; }

   const StatementList &classes() const { return m_classes; }
   StatementList &classes() { return m_classes; }

   void setExportAll( bool mode = true ) { m_exportAll = mode; }
   bool isExportAll() const { return m_exportAll; }

   SourceTree *clone() const;
};


}

#endif

/* end of syntree.h */

