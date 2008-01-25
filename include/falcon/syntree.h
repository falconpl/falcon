/*
   FALCON - The Falcon Programming Language.
   FILE: syntree.h
   $Id: syntree.h,v 1.19 2007/08/01 13:32:26 jonnymind Exp $

   Syntactic tree item definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven giu 4 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

public:
   RangeDecl( Value *start, Value *end = 0 ):
      m_rstart( start ),
      m_rend( end )
   {}

   RangeDecl( const RangeDecl &other );

   ~RangeDecl();

   bool isOpen() const { return m_rend == 0; }
   Value *rangeStart() const { return m_rstart; }
   Value *rangeEnd() const { return m_rend; }
   RangeDecl *clone() const { return new RangeDecl( *this ); }
};




class FALCON_DYN_CLASS Value: public BaseAlloc
{
public:
   typedef enum {
      t_nil,
      t_imm_integer,
      t_imm_string,
      t_imm_num,
      t_symbol,
      t_symdef,
      t_self,
      t_sender,

      t_byref,
      t_array_decl,
      t_dict_decl,
      t_range_decl,

      t_expression
   } type_t;

private:
   type_t m_type;
   union {
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

   /** Creates a Falcon::VarDef using the contents of this object.
      This version will create a VarDef also for symbols and references.
      \return a newly allocated prodef that has the same contents a the value, or 0
         for complex values.
   */
   VarDef *genVarDefSym();

   bool isImmediate() const {
      return m_type == t_nil ||
             m_type == t_imm_integer ||
             m_type == t_imm_string ||
             m_type == t_imm_num;
   }

   bool isSimple() const {
      return isImmediate() ||
             m_type == t_symbol || m_type == t_symdef ||
             m_type == t_self || m_type == t_sender;
   }

   bool isTrue() const {
      switch( m_type ) {
         case t_imm_integer: return asInteger() != 0;
         case t_imm_num:  return asNumeric() != 0.0;
         case t_imm_string: return asString()->size() != 0;
      }
      return false;
   }

   int64 asInteger() const { return m_content.asInteger; }
   numeric asNumeric() const { return m_content.asNumeric; }
   String *asString() const { return m_content.asString; }
   String *asSymdef() const { return m_content.asString; }
   Symbol *asSymbol() const { return m_content.asSymbol; }

   Value *asReference() const { return m_content.asRef; }
   ArrayDecl *asArray() const { return m_content.asArray; }
   DictDecl *asDict() const { return m_content.asDict; }
   RangeDecl *asRange() const { return m_content.asRange; }
   Expression *asExpr() const { return m_content.asExpr; }

   void setNil() { m_type = t_nil; }
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
   void setSender() { m_type = t_sender; }


   bool isNil() const {return m_type == t_nil; }
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
   bool isSender() const { return m_type == t_sender; }
   bool isRange() const { return m_type == t_range_decl; }

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
   extern ValuePtrTraits t_valueptr;
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
      t_neq,

      t_has,
      t_hasnt,
      t_in,
      t_notin,
      t_provides,

      t_iif,
      t_let,
      t_lambda,

      t_obj_access,
      t_funcall,
      t_funval,
      t_inherit,
      t_array_access,
      t_array_byte_access,
      t_strexpand,
      t_indirect,
      /** An optimized expression is like an unary operator */
      t_optimized,
   } operator_t;

private:
   operator_t m_operator;
   Value *m_first;
   Value *m_second;
   Value *m_third;

public:
   Expression( operator_t t, Value *first, Value *second = 0, Value *third = 0 ):
      m_first( first ),
      m_second( second ),
      m_third( third ),
      m_operator( t )
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
      t_assignment,
      t_autoadd,
      t_autosub,
      t_automul,
      t_autodiv,
      t_automod,
      t_autopow,
      t_autoband,
      t_autobor,
      t_autobxor,
      t_autoshl,
      t_autoshr,
      t_if,
      t_elif,
      t_while,
      t_forin,
      t_for,
      t_try,
      t_catch,
      t_switch,
      t_select,
      t_case,
      t_module,
      t_global,
      t_pass,

      t_object,
      t_class,
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

   Statement *clone() const;
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

   virtual Statement *clone() const;
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
   virtual Statement *clone() const;
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

   virtual Statement *clone() const;

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
   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtFordot: public StmtExpression
{
public:
   StmtFordot( uint32 line, Value *exp ):
      StmtExpression( line, t_fordot, exp )
   {}

   StmtFordot( const StmtFordot &other );

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoexpr: public StmtExpression
{
public:
   StmtAutoexpr( uint32 line, Value *exp ):
      StmtExpression( line, t_autoexp, exp )
   {}

   StmtAutoexpr( const StmtAutoexpr &other );

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtPass: public Statement
{
   Value *m_called;
   Value *m_saveRetIn;

public:
   StmtPass( uint32 line, Value *exp, Value *in = 0 ):
      Statement( line, t_pass ),
      m_called( exp ),
      m_saveRetIn( in )
   {}

   StmtPass( const StmtPass &other );

   virtual ~StmtPass();

   Value *called() const { return m_called; }
   Value *saveIn() const { return m_saveRetIn; }

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtGive: public Statement
{
   Value *m_object;
   ArrayDecl *m_attribs;

public:
   StmtGive( uint32 line, Value *object, ArrayDecl *attribs ):
      Statement( line, t_give ),
      m_object( object ),
      m_attribs( attribs )
   {}

   StmtGive( const StmtGive &other );

   virtual ~StmtGive();

   Value *object() const { return m_object; }
   ArrayDecl *attributes() const { return m_attribs; }

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtBaseAssign: public StmtExpression
{
   Value *m_dest;

public:
   StmtBaseAssign( uint32 line, type_t t, Value *target, Value *expr ):
      StmtExpression( line, t, expr ),
      m_dest( target )
   {}

   StmtBaseAssign( const StmtBaseAssign &other );

   virtual ~StmtBaseAssign();

   Value *destination() const { return m_dest; }

   // base assign is a pure virtual class; no implementation for clone().
};


class FALCON_DYN_CLASS StmtAssignment: public StmtBaseAssign
{

public:
   StmtAssignment( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_assignment, target, expr )
   {}

   StmtAssignment( const StmtAssignment &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoAdd: public StmtBaseAssign
{
public:
   StmtAutoAdd( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autoadd, target, expr )
   {}

   StmtAutoAdd( const StmtAutoAdd &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoSub: public StmtBaseAssign
{
public:
   StmtAutoSub( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autosub, target, expr )
   {}

   StmtAutoSub( const StmtAutoSub &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoMul: public StmtBaseAssign
{
public:
   StmtAutoMul( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_automul, target, expr )
   {}

   StmtAutoMul( const StmtAutoMul &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoDiv: public StmtBaseAssign
{
public:
   StmtAutoDiv( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autodiv, target, expr )
   {}

   StmtAutoDiv( const StmtAutoDiv &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtAutoMod: public StmtBaseAssign
{
public:
   StmtAutoMod( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_automod, target, expr )
   {}

   StmtAutoMod( const StmtAutoMod &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoPow: public StmtBaseAssign
{
public:
   StmtAutoPow( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autopow, target, expr )
   {}

   StmtAutoPow( const StmtAutoPow &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoBAND: public StmtBaseAssign
{
public:
   StmtAutoBAND( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autoband, target, expr )
   {}

   StmtAutoBAND( const StmtAutoBAND &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtAutoBOR: public StmtBaseAssign
{
public:
   StmtAutoBOR( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autobor, target, expr )
   {}

   StmtAutoBOR( const StmtAutoBOR &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtAutoBXOR: public StmtBaseAssign
{
public:
   StmtAutoBXOR( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autobxor, target, expr )
   {}

   StmtAutoBXOR( const StmtAutoBXOR &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoSHL: public StmtBaseAssign
{
public:
   StmtAutoSHL( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autoshl, target, expr )
   {}

   StmtAutoSHL( const StmtAutoSHL &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtAutoSHR: public StmtBaseAssign
{
public:
   StmtAutoSHR( uint32 line, Value *target, Value *expr ):
      StmtBaseAssign( line, t_autoshr, target, expr )
   {}

   StmtAutoSHR( const StmtAutoSHR &other ):
      StmtBaseAssign( other )
   {}

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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
   virtual Statement *clone() const;
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


class FALCON_DYN_CLASS StmtWhile: public StmtConditional
{
public:
   StmtWhile( uint32 line, Value *cond ):
      StmtConditional( line, t_while, cond )
   {}

   StmtWhile( const StmtWhile &other ):
      StmtConditional( other )
   {}

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtForin: public StmtBlock
{
   StatementList m_first;
   StatementList m_last;
   StatementList m_middle;

   Value *m_source;
   Value *m_dest;

public:
   StmtForin( uint32 line, Value *dest, Value *source ):
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
   Value *dest() const { return m_dest; }

   virtual Statement *clone() const;
};


class FALCON_DYN_CLASS StmtFor: public StmtBlock
{
   Value *m_counter;
   Value *m_from;
   Value *m_to;
   Value *m_step;

public:
   StmtFor( uint32 line, Value *counter, Value *from, Value *to, Value *step = 0 ):
      StmtBlock( line, t_for ),
      m_counter( counter ),
      m_from( from ),
      m_to( to ),
      m_step( step )
   {}

   StmtFor( const StmtFor &other );

   virtual ~StmtFor();

   Value *counter() const { return m_counter; }
   Value *from() const { return m_from; }
   Value *to() const { return m_to; }
   Value *step() const { return m_step; }

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
};


/** Statement select.

*/
class FALCON_DYN_CLASS StmtSelect: public StmtSwitch
{
public:
   StmtSelect( uint32 line, Value *expr );
   StmtSelect( const StmtSelect &other );

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
};

/** Module statement.
   Actually, this holds an "unnamed block", which may be top-level or anything.
*/
/*class FALCON_DYN_CLASS StmtModule: public StmtBlock
{
public:
   StmtModule( uint32 line = 0):
      StmtBlock( line, t_module )
   {}

   virtual Statement *clone() const;
};*/



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

class FALCON_DYN_CLASS StmtClass: public StmtCallable
{
   StmtFunction *m_ctor;
   bool m_initGiven;
   /** if this class is a clone it must delete it's constructor statement. */
   bool m_bDeleteCtor;

public:

   StmtClass( uint32 line, Symbol *name ):
      StmtCallable( line, t_class, name ),
      m_initGiven( false ),
      m_bDeleteCtor( false ),
      m_ctor(0)
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

   virtual Statement *clone() const;
};

class FALCON_DYN_CLASS StmtFunction: public StmtCallable
{

private:
   StatementList m_staticBlock;
   StatementList m_statements;

   int m_lambda_id;
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

   virtual Statement *clone() const;
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

   virtual Statement *clone() const;
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

