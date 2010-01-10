/*
   FALCON - The Falcon Programming Language.
   FILE: syntree.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/syntree.h>
#include <falcon/compiler.h>

namespace Falcon
{

//================================================
// Array declaration
//

static void s_valueDeletor( void *data )
{
   Value *value = (Value *) data;
   delete value;
}

ArrayDecl::ArrayDecl():
   List( s_valueDeletor )
{}

ArrayDecl::ArrayDecl( const ArrayDecl &other ):
   List( s_valueDeletor )
{
   ListElement *element = other.begin();
   while( element != 0 ) {
      Value *elem = (Value *) element->data();
      pushBack( new Value( *elem ) );
      element = element->next();
   }
}


//================================================
// Dictionary declaration
//

static void s_valuePairDeletor( void *data )
{
   DictDecl::pair *p = (DictDecl::pair *) data;
   delete p->first;
   delete p->second;
   delete p;
}

DictDecl::DictDecl():
   List( s_valuePairDeletor )
{
}

DictDecl::DictDecl( const DictDecl &other ):
   List( s_valuePairDeletor )
{
   ListElement *iter = other.begin();

   while( iter != 0 ) {
      pair *elem = (pair *) iter->data();
      Value *first = new Value( *elem->first );
      Value *second = new Value( *elem->second );
      pushBack( first, second );
      iter = iter->next();
   }
}

void DictDecl::pushBack( Value *first, Value *second )
{
   pair *p = new pair;
   p->first = first;
   p->second = second;
   List::pushBack( p );
}


//================================================
// Range declaration
//


RangeDecl::RangeDecl( const RangeDecl &other )
{
   if ( other.m_rstart == 0 )
      m_rstart = 0;
   else
      m_rstart = new Value( *other.m_rstart );

   if ( other.m_rend == 0 )
      m_rend = 0;
   else
      m_rend = new Value( *other.m_rend );

   if ( other.m_step == 0 )
      m_step = 0;
   else
      m_step = new Value( *other.m_step );

}

RangeDecl::~RangeDecl()
{
   delete m_rstart;
   delete m_rend;
   delete m_step;
}

//================================================
// Value
//

Value *Value::clone() const
{
   return new Value( *this );
}

bool Value::isEqualByValue( const Value &other ) const
{
   if ( type() == other.type() )
   {
      switch( type() )
      {
         case Value::t_nil: return true;

         case Value::t_imm_integer:
            if( asInteger() == other.asInteger() )
               return true;
         break;

         case Value::t_range_decl:
            if( other.asRange()->rangeStart() == asRange()->rangeStart() &&
                  ( (asRange()->isOpen() && other.asRange()->isOpen()) ||
                    (! asRange()->isOpen() && ! other.asRange()->isOpen()
                       && other.asRange()->rangeEnd() == asRange()->rangeEnd())
                  )
               )
               return true;
         break;

         case Value::t_imm_num:
            if( asNumeric() == other.asNumeric() )
               return true;
         break;

         case Value::t_imm_string:
            if( *asString() == *other.asString() )
               return true;
         break;

         case Value::t_lbind:
            if( *asLBind() == *other.asLBind() )
               return true;
         break;

         case Value::t_symbol:
            if( asSymbol()->id() == other.asSymbol()->id() )
               return true;
         break;

         case Value::t_symdef:
            if( asSymdef() == other.asSymdef() )
               return true;
         break;

         default:
            break;
      }
   }

   return false;
}

bool Value::less( const Value &other ) const
{
   if ( type() == other.type() )
   {
      switch( type() )
      {
         case Value::t_nil: return true;

         case Value::t_imm_integer:
            if( asInteger() < other.asInteger() )
               return true;
         break;

         case Value::t_range_decl:
            if( asRange()->rangeStart()->asInteger() < other.asRange()->rangeStart()->asInteger() )
               return true;
            if ( ! asRange()->isOpen()  && ! other.asRange()->isOpen() && asRange()->rangeEnd()->asInteger() < other.asRange()->rangeEnd()->asInteger() )
               return true;
         break;

         case Value::t_imm_num:
            if( asNumeric() < other.asNumeric() )
               return true;
         break;

         case Value::t_imm_string:
            if( *asString() < *other.asString() )
               return true;
         break;

         case Value::t_lbind:
            if( *asLBind() < *other.asLBind() )
               return true;
         break;

         case Value::t_symbol:
            if( asSymbol()->id() < other.asSymbol()->id() )
               return true;
         break;

         case Value::t_symdef:
            if( *asSymdef() < *other.asSymdef() )
               return true;
         break;

         default:
            break;
      }
   }
   else {
      return ((uint32)type()) < ((uint32)other.type());
   }

   return false;
}


VarDef *Value::genVarDef()
{
   VarDef *def;

   switch( type() )
   {
      case Falcon::Value::t_nil:
         def = new Falcon::VarDef();
      break;

      case Falcon::Value::t_imm_integer:
         def = new Falcon::VarDef( (int64) asInteger() );
      break;

      case Falcon::Value::t_imm_num:
         def = new Falcon::VarDef( asNumeric() );
      break;

      case Falcon::Value::t_imm_string:
         def = new Falcon::VarDef( asString() );
      break;

      case Falcon::Value::t_imm_bool:
         def = new Falcon::VarDef( asBool() );
      break;

      default:
         def = 0;  // set a nil expression
   }

   return def;
}

Value::~Value()
{
   switch( m_type )
   {

      case t_byref:
         delete m_content.asRef;
      break;

      case t_array_decl:
         delete m_content.asArray;
      break;

      case t_dict_decl:
          delete m_content.asDict;
      break;

      case t_range_decl:
         delete m_content.asRange;
      break;

      case t_expression:
         delete m_content.asExpr;
      break;

      default:
      // In every other case, there is nothing to do as strings and symbols are held in the
      // module and are not to be disposed here.
         break;
   }

}


void Value::copy( const Value &other )
{
   m_type = other.m_type;

   switch( m_type )
   {

      case t_byref:
         m_content.asRef = new Value( *other.m_content.asRef );
      break;

      case t_array_decl:
         m_content.asArray = new ArrayDecl( *other.m_content.asArray );
      break;

      case t_dict_decl:
         m_content.asDict = new DictDecl( *other.m_content.asDict );
      break;

      case t_range_decl:
         m_content.asRange = new RangeDecl( *other.m_content.asRange );
      break;

      case t_expression:
         m_content.asExpr = new Expression( *other.m_content.asExpr );
      break;

      // In every other case, a flat copy is ok
      default:
         m_content = other.m_content;
   }
}

//================================================
// (owned) Value pointer traits for maps
//

void ValuePtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   Value **target = (Value**) targetZone;
   *target = ((Value *) sourceZone);
}


int ValuePtrTraits::compare( const void *firstArea, const void *secondv ) const
{
   Value *first = *(Value **) firstArea;
   Value *second = (Value *) secondv;

   if ( *first < *second )
      return -1;
   else if ( *first == *second )
      return 0;

   return 1;
}

void ValuePtrTraits::destroy( void *item ) const
{
   Value *v = *(Value **) item;
   delete v;
}

bool ValuePtrTraits::owning() const
{
   return true;
}


namespace traits {
   ValuePtrTraits &t_valueptr() { static ValuePtrTraits dt; return dt; }
}

//================================================
// Expression
//

Expression::Expression( const Expression &other )
{
   m_operator = other.m_operator;
   m_first = other.m_first == 0 ? 0 : new Value( *other.m_first );
   m_second = other.m_second == 0 ? 0 : new Value( *other.m_second );
   m_third = other.m_third == 0 ? 0 : new Value( *other.m_third );
}

Expression::~Expression()
{
   delete m_first;
   delete m_second;
   delete m_third;
}

bool Expression::isStandAlone() const
{
   switch( m_operator )
   {
      case t_eval: case t_funcall: case t_assign:
      case t_aadd: case t_asub: case t_amul: case t_adiv:
      case t_amod: case t_apow:
      case t_aband: case t_abor: case t_abxor:
      case t_ashl: case t_ashr:
      case t_pre_inc: case t_post_inc: case t_pre_dec: case t_post_dec:
         return true;

      // and are or are meaningful if their second element is an effective expression.
      // the first one needs not (i.e. may be a variable), and if the first one is
      // but the second one is not, then the second part does nothing, and we're
      // right in raising an error as there's something wrong.
      case t_and: case t_or:
         return m_second->isExpr() && m_second->asExpr()->isStandAlone();

      default:
         return false;
   }
}

//================================================
// Statement list
//
StatementList::StatementList( const StatementList &other )
{
   Statement *elem = other.front();
   while( elem != 0 ) {
      push_back( elem->clone() );
      elem = (Statement *) elem->next();
   }
}

StatementList::~StatementList()
{
   Statement *elem = pop_back();
   while( elem != 0 ) {
      delete elem;
      elem = pop_back();
   }
}

SymbolList::SymbolList( const SymbolList &other )
{
   // we don't have the ownership of the symbol;
   // we just need to duplicate the things.

   ListElement *elem = other.begin();
   while( elem != 0 )
   {
      pushBack( elem->data() );
      elem = elem->next();
   }
}

//================================================
// Statement list
//

Statement::Statement( const Statement &other ):
   m_type( other.m_type ),
   m_line( other.m_line )
{}


//================================================
// Statement NONE
//

StmtNone *StmtNone::clone() const
{
   return new StmtNone( *this );
}

// Global statement
//

StmtGlobal::StmtGlobal( const StmtGlobal &other ):
   Statement( other ),
   m_symbols( other.m_symbols )
{}

StmtGlobal *StmtGlobal::clone() const
{
   return new StmtGlobal( *this );
}

// StmtUnref statement
//

StmtUnref::StmtUnref( const StmtUnref &other ):
   Statement( other )
{
    m_symbol = other.m_symbol == 0 ? 0 : other.m_symbol->clone();
}

StmtUnref::~StmtUnref()
{
   delete m_symbol;
}

StmtUnref *StmtUnref::clone() const
{
   return new StmtUnref( *this );
}



// StmtSelfPrint statement
//

StmtSelfPrint::StmtSelfPrint( const StmtSelfPrint &other ):
   Statement( other ),
   m_toPrint( new ArrayDecl( *other.m_toPrint ) )
{}

StmtSelfPrint::~StmtSelfPrint()
{
   delete m_toPrint;
}

StmtSelfPrint *StmtSelfPrint::clone() const
{
   return new StmtSelfPrint( *this );
}


// StmtExpression statement
//

StmtExpression::StmtExpression( const StmtExpression &other ):
   Statement( other )
{
   m_expr = other.m_expr == 0 ? 0 : other.m_expr->clone();
}

StmtExpression::~StmtExpression()
{
   delete m_expr;
}

StmtExpression *StmtExpression::clone() const
{
   return new StmtExpression( *this );
}

// StmtAutoexpr statement
//

StmtAutoexpr::StmtAutoexpr( const StmtAutoexpr &other ):
   StmtExpression( other )
{}

StmtAutoexpr *StmtAutoexpr::clone() const
{
   return new StmtAutoexpr( *this );
}

// StmtFordot statement
//

StmtFordot::StmtFordot( const StmtFordot &other ):
   StmtExpression( other )
{}

StmtFordot *StmtFordot::clone() const
{
   return new StmtFordot( *this );
}

// StmtReturn statement
//

StmtReturn *StmtReturn::clone() const
{
   return new StmtReturn( *this );
}

// StmtRaise statement
//

StmtRaise *StmtRaise::clone() const
{
   return new StmtRaise( *this );
}

// StmtLaunch statement
//

StmtLaunch *StmtLaunch::clone() const
{
   return new StmtLaunch( *this );
}


// StmtGive statement
//


StmtGive::StmtGive( const StmtGive &other ):
   Statement( other )
{
   m_objects = other.m_objects == 0 ? 0 : new ArrayDecl( *other.m_objects );
   m_attribs = other.m_attribs == 0 ? 0 : new ArrayDecl( *other.m_attribs );
}

StmtGive::~StmtGive()
{
   delete m_objects;
   delete m_attribs;
}

StmtGive *StmtGive::clone() const
{
   return new StmtGive( *this );
}

// Loop ctrl
//

StmtLoopCtl::StmtLoopCtl( const StmtLoopCtl &other ):
   Statement( other )
{
}

StmtBreak *StmtBreak::clone() const
{
   return new StmtBreak( *this );
}

StmtContinue *StmtContinue::clone() const
{
   return new StmtContinue( *this );
}

// StmtBlock statement
//

StmtBlock::StmtBlock( const StmtBlock &other ):
   Statement( other ),
   m_list( other.m_list )
{
}

// StmtConditional statement
//

StmtConditional::StmtConditional( const StmtConditional &other ):
   StmtBlock( other )
{
   m_condition = other.m_condition == 0 ? 0 : new Value( *other.m_condition );
}

StmtConditional::~StmtConditional()
{
   delete m_condition;
}

// StmtLoop statement
//
StmtLoop *StmtLoop::clone() const
{
   return new StmtLoop( *this );
}

// StmtWhile statement
//
StmtWhile *StmtWhile::clone() const
{
   return new StmtWhile( *this );
}

// StmtElif statement
//

StmtElif *StmtElif::clone() const
{
   return new StmtElif( *this );
}

// Stmtif statement
//

StmtIf::StmtIf( const StmtIf &other ):
   StmtConditional( other ),
   m_else( other.m_else ),
   m_elseifs( other.m_elseifs )
{
}

StmtIf *StmtIf::clone() const
{
   return new StmtIf( *this );
}

// StmtForin statement
//

StmtForin::StmtForin( const StmtForin &other ):
   StmtBlock( other ),
   m_first( other.m_first ),
   m_last( other.m_last ),
   m_middle( other.m_middle )
{
   m_source = other.m_source == 0 ? 0 : new Value( * other.m_source );
   m_dest = other.m_dest == 0 ? 0 : new ArrayDecl( * other.m_dest );
}

StmtForin::~StmtForin()
{
   delete m_source;
   delete m_dest;
}

StmtForin *StmtForin::clone() const
{
   return new StmtForin( *this );
}


// StmtCaseBlock statement
//

StmtCaseBlock *StmtCaseBlock::clone() const
{
   return new StmtCaseBlock( *this );
}

// StmtSwitch statement
//

StmtSwitch::StmtSwitch( uint32 line, Value *expr ):
   Statement( line, t_switch ),
   m_cases_int( &traits::t_valueptr(), &traits::t_int(), 19 ),
   m_cases_rng( &traits::t_valueptr(), &traits::t_int(), 19 ),
   m_cases_str( &traits::t_valueptr(), &traits::t_int(), 19 ),
   m_cases_obj( &traits::t_valueptr(), &traits::t_int(), 19 ),
   m_nilBlock( -1 ),
   m_cfr( expr )
{}

StmtSwitch::StmtSwitch( const StmtSwitch &other ):
   Statement( other ),
   m_cases_int( other.m_cases_int ),
   m_cases_rng( other.m_cases_rng ),
   m_cases_str( other.m_cases_str ),
   m_cases_obj( other.m_cases_obj ),
   m_obj_list( other.m_obj_list ),
   m_blocks( other.m_blocks ),
   m_defaultBlock( other.m_defaultBlock ),
   m_nilBlock( other.m_nilBlock )
{
   m_cfr = other.m_cfr == 0 ? 0 : new Value( *other.m_cfr );
}

StmtSwitch::~StmtSwitch()
{

   delete m_cfr;
}

bool StmtSwitch::addIntCase( Value *itm )
{
   if ( m_cases_int.find( itm ) != 0 ) {
      return false;
   }

   // check against duplicated cases in ranges
   int32 val = (int32) itm->asInteger();
   MapIterator iter = m_cases_rng.begin();
   while( iter.hasCurrent() )
   {
      Value *rng = *(Value **) iter.currentKey();
      Value *begin = rng->asRange()->rangeStart();

      if ( rng->asRange()->isOpen() )
      {
         if ( val >= (int32) begin->asInteger() )
            return false;
      }

      Value *end = rng->asRange()->rangeEnd();
      if ( val >= (int32) begin->asInteger()  && val <= (int32) end->asInteger())
         return false;

      iter.next();
   }

   uint32 temp = m_blocks.size();
   m_cases_int.insert( itm, &temp );

   return true;
}

bool StmtSwitch::addStringCase( Value *itm )
{
   if ( m_cases_str.find( itm ) != 0 ) {
      return false;
   }

   uint32 temp = m_blocks.size();
   m_cases_str.insert( itm, &temp );
   return true;
}

bool StmtSwitch::addRangeCase( Value *itm )
{
   if ( m_cases_rng.find( itm ) != 0 ) {
      return false;
   }

   // todo check int and ranges
   // check against duplicated cases in ranges
   bool isOpen = itm->asRange()->isOpen();
   int32 start = (int32) itm->asRange()->rangeStart()->asInteger();
   int32 end = isOpen ? 0 : (int32) itm->asRange()->rangeEnd()->asInteger();

   // only positive range intervals are meaningful
   if ( ! isOpen && end < start )
   {
      itm->asRange()->rangeStart()->setInteger( end );
      itm->asRange()->rangeEnd()->setInteger( start );
      int32 t = start;
      start = end;
      end = t;
   }

   // check integers
   MapIterator iter = m_cases_int.begin();
   while( iter.hasCurrent() )
   {
      Value *first = *(Value **) iter.currentKey();
      int32 val = (int32) first->asInteger();

      if ( (isOpen && val >= start) || (!isOpen && val >= start && val <= end) )
      {
         return false;
      }

      iter.next();
   }


   // then check against ranges
   iter = m_cases_rng.begin();
   while( iter.hasCurrent() )
   {
      Value *rng = *(Value **) iter.currentKey();
      // as all the inserted ranges have been forced to positive interval
      // there's no need to do it here again.
      bool other_isOpen = rng->asRange()->isOpen();
      int other_start = (int) (rng->asRange()->rangeStart()->asInteger());
      int other_end = (int) (other_isOpen ? 0 : rng->asRange()->rangeEnd()->asInteger());
      if ( other_isOpen )
      {
         if ( isOpen )
            return false;

         if ( end >= other_start )
            return false;
      }
      else {
         if ( isOpen )
         {
            if  ( other_end >= other_end )
               return false;
         }
         else
         {
            if ( (start <= other_start && other_start <= end) ||
                 (start <= other_end && other_end <= end) ||
                 (other_start <= start && start <= other_end) ||
                 (other_start <= end && end <= other_end)
            )
               return false;
         }
      }

      iter.next();
   }

   // fine.
   uint32 temp = m_blocks.size();
   m_cases_rng.insert( itm, &temp );
   return true;
}

bool StmtSwitch::addSymbolCase( Value *itm )
{
   if ( m_cases_obj.find( itm ) != 0 ) {
      return false;
   }

   uint32 temp = m_blocks.size();
   m_cases_obj.insert( itm, &temp );
   m_obj_list.pushBack( itm );
   return true;
}

void StmtSwitch::addBlock( StmtCaseBlock *sl )
{
   m_blocks.push_back( sl );
}

StmtSwitch *StmtSwitch::clone() const
{
   return new StmtSwitch( *this );
}


// StmtSelect statement
//

StmtSelect::StmtSelect( uint32 line, Value *expr ):
   StmtSwitch( line, expr )
{
   m_type = t_select;
}

StmtSelect::StmtSelect( const StmtSelect &other ):
   StmtSwitch( other )
{
   m_type = t_select;
}

StmtSelect *StmtSelect::clone() const
{
   return new StmtSelect( *this );
}

// StmtCatchBlock statement
//

StmtCatchBlock::StmtCatchBlock( const StmtCatchBlock &other ):
   StmtBlock( other )
{
   m_into = other.m_into == 0 ? 0 : new Value( *other.m_into );
}


StmtCatchBlock::~StmtCatchBlock()
{
   delete m_into;
}

StmtCatchBlock *StmtCatchBlock::clone() const
{
   return new StmtCatchBlock( *this );
}

// StmtCatchBlock statement
//

StmtTry::StmtTry( uint32 line ):
   StmtBlock( line, t_try ),
   m_cases_int( &traits::t_valueptr(), &traits::t_int(), 13 ),
   m_cases_sym( &traits::t_valueptr(), &traits::t_int(), 13 ),
   m_into_values( s_valueDeletor ),
   m_default(0)
{}

StmtTry::StmtTry( const StmtTry &other ):
   StmtBlock( other ),
   m_cases_int( other.m_cases_int ),
   m_cases_sym( other.m_cases_sym ),
   m_sym_list( other.m_sym_list ),
   m_handlers( other.m_handlers ),
   m_into_values( s_valueDeletor )
{
   m_default = other.m_default == 0 ? 0 : other.m_default;
   ListElement *elem = other.m_into_values.begin();
   while( elem != 0 )
   {
      Value *into = (Value *) elem->data();
      m_into_values.pushBack( new Value( *into ) );
      elem = elem->next();
   }

}

StmtTry::~StmtTry()
{
   delete m_default;
}

void StmtTry::defaultHandler( StmtCatchBlock *block )
{
   delete m_default;
   m_default = block;
}

void StmtTry::addHandler( StmtCatchBlock *sl )
{
   m_handlers.push_back( sl );
}

bool StmtTry::addIntCase( Value *itm )
{
   if ( m_cases_int.find( itm ) != 0 ) {
      return false;
   }

   uint32 temp = m_handlers.size();
   m_cases_int.insert( itm, &temp );

   return true;
}

bool StmtTry::addSymbolCase( Value *itm )
{
   if ( m_cases_sym.find( itm ) != 0 ) {
      return false;
   }

   uint32 temp = m_handlers.size();
   m_cases_sym.insert( itm, &temp );
   m_sym_list.pushBack( itm );
   return true;
}

StmtTry *StmtTry::clone() const
{
   return new StmtTry( *this );
}

// StmtCallable statement
//

StmtCallable::~StmtCallable()
{}

// StmtClass statement
//

StmtClass::StmtClass( const StmtClass &other ):
   StmtCallable( other ),
   m_initGiven( other.m_initGiven ),
   m_states( other.m_states )
{
   if ( other.m_ctor == 0 )
   {
      m_ctor = 0;
      m_bDeleteCtor = false;
   }
   else {
      new StmtFunction( *other.m_ctor );
      m_bDeleteCtor = true;
   }

   m_ctor->setConstructorFor( this );

   if( other.m_initState != 0 )
      m_initState = other.m_initState->clone();
   else
      m_initState = 0;
}


StmtClass::~StmtClass()
{
   if( m_bDeleteCtor )
      delete m_ctor;
}

bool StmtClass::addState( StmtState* state )
{
   Statement* f = m_states.front();
   while( f != 0 )
   {
      StmtState* fs = static_cast<StmtState*>( f );
      if ( * fs->name() == *state->name() )
         return false;
      f = static_cast<Statement*>( f->next() );
   }

   m_states.push_back( state );
   return true;
}

StmtClass *StmtClass::clone() const
{
   return new StmtClass( *this );
}

//===================================================
// State statement
//

StmtState::StmtState( const String* name, StmtClass* owner ):
      Statement( t_state ),
      m_name( name ),
      m_owner( owner ),
      m_funcs( &traits::t_stringptr(), &traits::t_voidp(), 19 )
{
   m_stateDef = new StateDef( name );
}


StmtState::~StmtState()
{
}

StmtState::StmtState( const StmtState& other ):
   Statement( other ),
   m_name( other.m_name ),
   m_owner( other.m_owner ),
   m_funcs( other.m_funcs )
{
   m_stateDef = new StateDef( other.m_name );
}

StmtState* StmtState::clone() const
{
   return new StmtState( *this );
}


bool StmtState::addFunction( const String* name, Symbol* func )
{
   if( m_funcs.find( name ) != 0 )
      return false;

   m_funcs.insert( name, func );
   return true;
}



//===================================================
// StmtFunction statement
//

StmtFunction::StmtFunction( const StmtFunction &other ):
   StmtCallable( other ),
   m_lambda_id( other.m_lambda_id ),
   m_staticBlock( other.m_staticBlock ),
   m_statements( other.m_statements ),
   m_ctor_for( 0 )  // ctor for always zero; it's eventually set by the class copy constructor
{
}

StmtFunction *StmtFunction::clone() const
{
   return new StmtFunction( *this );
}

// StmtFunction statement
//

StmtVarDef::StmtVarDef( const StmtVarDef &other ):
   Statement( other ),
   m_name( other.m_name ) // the string is in the module
{
   m_value = other.m_value == 0 ? 0 : new Value( *other.m_value );
}

StmtVarDef::~StmtVarDef()
{
   delete m_value;
}

StmtVarDef *StmtVarDef::clone() const
{
   return new StmtVarDef( *this );
}

// StmtFunction statement
//

SourceTree::SourceTree( const SourceTree &other ):
   m_statements( other.m_statements ),
   m_functions( other.m_functions ),
   m_classes( other.m_classes ),
   m_exportAll( other.m_exportAll )
{
}

SourceTree *SourceTree::clone() const
{
   return new SourceTree( *this );
}

}

/* end of syntree.cpp */
