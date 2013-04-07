/*
   FALCON - The Falcon Programming Language.
   FILE: classstring.cpp

   String type handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classstring.cpp"

#include <falcon/classes/classstring.h>
#include <falcon/range.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/errors/accesserror.h>
#include <falcon/errors/operanderror.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

namespace Falcon {

   
//=====================================================================
// Properties
//

static void get_len( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<String*>( instance )->length();
}

static void get_isText( const Class*, const String&, void* instance, Item& value )
{
   value.setBoolean( static_cast<String*>( instance )->isText() );
}

static void set_isText( const Class*, const String&, void* instance, const Item& value )
{
   String* str = static_cast<String*>( instance );
   if( value.isTrue() ) {
      if( ! str->isText() ) {
         str->manipulator( str->manipulator()->bufferedManipulator() );
      }
   }
   else {
      str->toMemBuf();
   }
}

//
// Class properties used for enumeration
//

ClassString::ClassString():
   Class( "String", FLC_CLASS_ID_STRING ),
   m_nextOp(this)
{
   addProperty( "isText", &get_isText, &set_isText );
   addProperty( "len", &get_len );
}


ClassString::~ClassString()
{
}

int64 ClassString::occupiedMemory( void* instance ) const
{
   /* NO LOCK */
   String* s = static_cast<String*>( instance );
   return sizeof(String) + s->allocated() + 16 + (s->allocated()?16:0);
}


void ClassString::dispose( void* self ) const
{
   /* NO LOCK */
   delete static_cast<String*>( self );
}


void* ClassString::clone( void* source ) const
{
   String* temp;
   {
      InstanceLock::Locker( &m_lock, source );
      temp = new String( *( static_cast<String*>( source ) ) );
   }

   return temp;
}

void* ClassString::createInstance() const
{
   return new String;
}

void ClassString::store( VMContext*, DataWriter* dw, void* data ) const
{
#ifdef FALCON_MT_UNSAFE
   String& value = *static_cast<String*>( data );
   TRACE2( "ClassString::store -- (unsafe) \"%s\"", value.c_ize() );
#else
   InstanceLock::Token* tk = m_lock.lock(data);
   String value(*static_cast<String*>( data ));
   m_lock.unlock(tk);

   TRACE2( "ClassString::store -- \"%s\"", value.c_ize() );
#endif

   dw->write( value );
}


void ClassString::restore( VMContext* ctx, DataReader* dr ) const
{
   String* str = new String;

   try
   {
      dr->read( *str );
      TRACE2( "ClassString::restore -- \"%s\"", str->c_ize() );
      ctx->pushData( Item( this, str ) );
   }
   catch( ... )
   {
      delete str;
      throw;
   }
}


void ClassString::describe( void* instance, String& target, int, int maxlen ) const
{
#ifdef FALCON_MT_UNSAFE
   String* self = static_cast<String*>( instance );
#else
   InstanceLock::Token* tk = m_lock.lock(instance);
   String copy( *static_cast<String*>( instance ) );
   m_lock.unlock(tk);

   String* self = &copy;
#endif

   target.size( 0 );

   if( self->isText() )
   {
      String escaped;
      self->escape(escaped);

      target.append( '"' );
      if ( (int) self->length() > maxlen && maxlen > 0 )
      {
         target.append( escaped.subString( 0, maxlen ) );
         target.append( "..." );
      }
      else
      {
         target.append( escaped );
      }
      target.append( '"' );
   }
   else {
      target.append( "m{" );

      length_t pos = 0;
      byte* data = self->getRawStorage();
      while( pos < self->size() && (maxlen <=0 || pos*3 < (unsigned int) maxlen) ) {
         if( pos > 0 ) target.append(' ');
         if( data[pos] < 16 )
         {
            target.append('0');
         }
         target.writeNumberHex( data[pos], true );
         ++pos;
      }

      target.append( '}' );
   }
}


void ClassString::gcMarkInstance( void* instance, uint32 mark ) const
{
   /* NO LOCK */
   static_cast<String*>( instance )->gcMark( mark );
}


bool ClassString::gcCheckInstance( void* instance, uint32 mark ) const
{
   /* NO LOCK */
   return static_cast<String*>( instance )->currentMark() >= mark;
}


//=======================================================================
// Addition

void ClassString::op_add( VMContext* ctx, void* self ) const
{
   String* str = static_cast<String*>( self );

   Item* op1, *op2;

   ctx->operands( op1, op2 );

   Class* cls;
   void* inst;

   if ( ! op2->asClassInst( cls, inst ) )
   {
      InstanceLock::Token* tk = m_lock.lock(str);
      String* copy = new String( *str );
      m_lock.unlock(tk);

      copy->append( op2->describe() );

      ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx,copy) );

      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      InstanceLock::Token* tk = m_lock.lock(str);
      String* copy = new String( *str );
      m_lock.unlock(tk);

      tk = m_lock.lock(inst);
      copy->append( *static_cast<String*>( inst ) );
      m_lock.unlock(tk);

      ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx, copy) );

      return;
   }

   // else we surrender, and we let the virtual system to find a way.
   ctx->pushCode( &m_nextOp );

   // this will transform op2 slot into its string representation.
   cls->op_toString( ctx, inst );

   if ( ! ctx->wentDeep( &m_nextOp ) )
   {
      ctx->popCode();

      // op2 has been transformed
      String* deep = (String*)op2->asInst();

      InstanceLock::Token* tk = m_lock.lock(str);
      deep->prepend( *str );
      m_lock.unlock(tk);
   }
}

//=======================================================================
// Operands
//

bool ClassString::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   String* self = static_cast<String*>(instance);

   // no param?
   if ( pcount > 0 )
   {
      // the parameter is a string?
      Item* itm = ctx->opcodeParams( pcount );

      if ( itm->isString() )
      {
         // copy it.
         self->copy( *itm->asString() );
      }
      else
      {
         if( pcount > 1 ) {
            ctx->popData( pcount-1 );
            // just to be sure.
            itm = &ctx->topData();
         }

         // apply the op_toString on the item.
         ctx->pushCode( &m_initNext );
         ctx->currentCode().m_seqId = pcount;
         long depth = ctx->codeDepth();

         // first get the required data...
         Class* cls;
         void* data;
         itm->forceClassInst( cls, data );

         // then ensure that the stack is as we need.
         ctx->pushData( *self );
         ctx->pushData( *itm );

         // and finally invoke stringify operation.
         cls->op_toString( ctx, data );
         if( depth == ctx->codeDepth() )
         {
            // we can get the string here.
            fassert( ctx->topData().isString() );
            fassert( ctx->opcodeParam(1).isString() );

            String* result = ctx->topData().asString();
            ctx->opcodeParam(1).asString()->copy( *result );

            // and clean the stack
            ctx->popData(2 + pcount);
            ctx->popCode();
         }

         // we took care of the stack.
         return true;
      }
   }

   return false;
}


void ClassString::op_aadd( VMContext* ctx, void* self ) const
{
   String* str = static_cast<String*>( self );

   Item* op1, *op2;
   ctx->operands( op1, op2 );

   Class* cls=0;
   void* inst=0;

   if ( op2->isString() )
   {
      if ( op1->copied() )
      {
         String* copy = new String;
         InstanceLock::Token* tk = m_lock.lock(str);
         copy->append( *str );
         m_lock.unlock(tk);

         tk = m_lock.lock(op2->asString());
         copy->append( *op2->asString() );
         m_lock.unlock(tk);
         ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx, copy) );
      }
      else
      {
#ifdef FALCON_MT_UNSAFE
         op1->asString()->append( *op2->asString() );
#else
         InstanceLock::Token* tk = m_lock.lock(op2->asString());
         String copy( *op2->asString() );
         m_lock.unlock(tk);

         tk = m_lock.lock(op1->asString());
         op1->asString()->append(copy);
         m_lock.unlock(tk);
#endif

         ctx->popData();
      }

      return;
   }
   else if ( ! op2->asClassInst( cls, inst ) )
   {
      // a flat entity
      if ( op1->copied() )
      {
         String* copy = new String;
         InstanceLock::Token* tk = m_lock.lock(str);
         copy->append( *str );
         m_lock.unlock(tk);

         copy->append( op2->describe() );
         ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx,copy) );
      }
      else
      {
         InstanceLock::Token* tk = m_lock.lock(op1->asString());
         op1->asString()->append( op2->describe() );
         m_lock.unlock(tk);
      }
      return;
   }


   // else we surrender, and we let the virtual system to find a way.
   ctx->pushCode( &m_nextOp );

   // this will transform op2 slot into its string representation.
   cls->op_toString( ctx, inst );

   if( ! ctx->wentDeep( &m_nextOp ) )
   {
      ctx->popCode();

      // op2 has been transformed (and is ours)
      String* deep = (String*) op2->asInst();

      InstanceLock::Token* tk = m_lock.lock(str);
      deep->prepend( *str );
      m_lock.unlock(tk);
   }
}


ClassString::NextOp::NextOp( ClassString* owner ):
         m_owner(owner)
{
   apply = apply_;
}


void ClassString::NextOp::apply_( const PStep* ps, VMContext* ctx )
{
   const ClassString::NextOp* step = static_cast<const ClassString::NextOp*>(ps);

   // The result of a deep call is in A
   Item* op1, *op2;

   ctx->operands( op1, op2 ); // we'll discard op2

   String* deep = op2->asString();
   String* self = op1->asString();

   if( op1->copied() )
   {
      String* copy = new String;
      InstanceLock::Token* tk = step->m_owner->m_lock.lock(self);
      copy->append( *self );
      step->m_owner->m_lock.unlock(tk);
      copy->append( *deep );
      ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx,copy) );
   }
   else
   {
      ctx->popData();
      InstanceLock::Token* tk = step->m_owner->m_lock.lock(self);
      self->append( *deep );
      step->m_owner->m_lock.unlock(tk);
   }

   ctx->popCode();
}


ClassString::InitNext::InitNext()
{
   apply = apply_;
}


void ClassString::InitNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->opcodeParam(1).asString()->copy( *ctx->topData().asString() );
   // remove the locally pushed data and the parameters.
   ctx->popData( 2 + ctx->currentCode().m_seqId );
   ctx->popCode();
}


//===============================================================
//

void ClassString::op_mul( VMContext* ctx, void* instance ) const
{
   // self count => new
   Item& i_count = ctx->topData();
   if( ! i_count.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "N" ) );
   }

   int64 count = i_count.forceInteger();
   ctx->popData();
   if( count == 0 )
   {
      ctx->topData() = FALCON_GC_HANDLE(new String);
      return;
   }

   String* self = static_cast<String*>(instance);
   InstanceLock::Token* tk = m_lock.lock(self);

   String copy(*self);
   m_lock.unlock(tk);

   String* result = new String;
   result->reserve(copy.size() * count);
   for( int64 i = 0; i < count; ++i )
   {
      result->append(copy);
   }

   ctx->topData() = FALCON_GC_HANDLE(result);
}


void ClassString::op_amul( VMContext* ctx, void* instance ) const
{
   // self count => self
   Item& i_count = ctx->topData();
   if( ! i_count.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "N" ) );
   }

   int64 count = i_count.forceInteger();
   ctx->popData();

   String* self = static_cast<String*>(instance);
   String* target;
   InstanceLock::Token* tk = m_lock.lock(self);

   String copy(*self);
   if( ctx->topData().copied() )
   {
      target = new String(copy);
      ctx->topData() = FALCON_GC_HANDLE(target);
      m_lock.unlock(tk);
      tk = 0;
   }
   else {
      target = self;
   }

   if( count == 0 )
   {
      target->size(0);
   }
   else
   {
      target->reserve( target->size() * count);
      // start from 1: we have already 1 copy in place
      for( int64 i = 1; i < count; ++i )
      {
         target->append(copy);
      }
   }

   if( tk != 0 )
   {
      m_lock.unlock(tk);
   }
}

void ClassString::op_div( VMContext* ctx, void* instance ) const
{
   // self count => new
   Item& i_count = ctx->topData();
   if( ! i_count.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "N" ) );
   }

   int64 count = i_count.forceInteger();
   ctx->popData();

   if ( count < 0 || count >= 0xFFFFFFFFLL )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "out of range" ) );
   }

   String* self = static_cast<String*>(instance);
   InstanceLock::Token* tk = m_lock.lock(self);
   String* target = new String(*self);
   target->append((char_t) count);
   ctx->topData() = FALCON_GC_HANDLE( target );
   m_lock.unlock(tk);
}


void ClassString::op_adiv( VMContext* ctx, void* instance ) const
{
   // self count => new
   Item& i_count = ctx->topData();
   if( ! i_count.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "N" ) );
   }

   int64 count = i_count.forceInteger();
   ctx->popData();

   if ( count < 0 || count >= 0xFFFFFFFFLL )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "out of range" ) );
   }

   String* self = static_cast<String*>(instance);
   InstanceLock::Token* tk = m_lock.lock(self);

   String* target;
   if( ctx->topData().copied() )
   {
      target = new String(*self);
      ctx->topData() = FALCON_GC_HANDLE(target);
   }
   else {
      target = self;
   }

   target->append((char_t) count);
   m_lock.unlock(tk);
}


void ClassString::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *stritem;

   ctx->operands( stritem, index );

   String& str = *static_cast<String*>( self );

   if ( index->isOrdinal() )
   {
      int64 v = index->forceInteger();
      uint32 chr = 0;

      {
         InstanceLock::Locker( &m_lock, &str );

         if ( v < 0 ) v = str.length() + v;

         if ( v >= str.length() )
         {
            throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
         }

         chr = str.getCharAt( (length_t) v );
      }

      if( str.isText() ) {
         String *s = new String();
         s->append( chr );
         ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx,s) );
      }
      else {
         ctx->stackResult(2, Item((int64) chr) );
      }
   }
   else if ( index->isUser() ) // index is a range
   {
      // if range is moving from a smaller number to larger (start left move right in the array)
      //      give values in same order as they appear in the array
      // if range is moving from a larger number to smaller (start right move left in the array)
      //      give values in reverse order as they appear in the array

      Class *cls;
      void *udata;

      if ( ! index->asClassInst( cls, udata ) )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "type error" ) );
      }

      // Confirm we have a range
      if ( cls->typeID() != FLC_CLASS_ID_RANGE )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "unknown index" ) );
      }

      Range& rng = *static_cast<Range*>( udata );

      int64 step = ( rng.step() == 0 ) ? 1 : rng.step(); // assume 1 if no step given
      int64 start = rng.start();
      int64 end = rng.end();

      bool reverse = false;
      String *s = new String();

      {
         InstanceLock::Locker( &m_lock, &str );
         int64 strLen = str.length();

         // do some validation checks before proceeding
         if ( start >= strLen || start < ( strLen * -1 )  || end > strLen || end < ( strLen * -1 ) )
         {
            delete s;
            throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
         }

         if ( rng.isOpen() )
         {
            // If negative number count from the end of the array
            if ( start < 0 ) start = strLen + start;

            end = strLen;
         }
         else // non-open range
         {
            if ( start < 0 ) start = strLen + start;

            if ( end < 0 ) end = strLen + end;

            if ( start > end )
            {
               reverse = true;
               if ( rng.step() == 0 ) step = -1;
            }
         }

         if ( reverse )
         {
            while ( start >= end )
            {
               s->append( str.getCharAt( (length_t) start ) );
               start += step;
            }
         }
         else
         {
            while ( start < end )
            {
               s->append( str.getCharAt( (length_t) start ) );
               start += step;
            }
         }

         if( ! str.isText() )
         {
            s->toMemBuf();
         }
      }

      ctx->stackResult( 2, FALCON_GC_HANDLE_IN(ctx,s) );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "invalid index" ) );
   }
}


void ClassString::op_setIndex( VMContext* ctx, void* self ) const
{
   Item* value, *arritem, *index;

   ctx->operands( value, arritem, index );

   String& str = *static_cast<String*>( self );

   if ( ! value->isString() && ! value->isOrdinal())
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("non string/char being assigned") );
   }

   if ( index->isOrdinal() )
   {
      // simple index assignment: a[x] = value
      {
         InstanceLock::Locker( &m_lock, &str );

         int64 v = index->forceInteger();

         if ( v < 0 ) v = str.length() + v;

         if ( v >= str.length() )
         {
            throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("index out of range") );
         }

         if( value->isOrdinal() ) {
            str.setCharAt( (length_t) v, (char_t) value->forceInteger() );
         }
         else {
            str.setCharAt( (length_t) v, (char_t) value->asString()->getCharAt( 0 ) );
         }
      }

      ctx->stackResult( 3, *value );
   }
   else if ( index->isRange() )
   {
      Range& rng = *static_cast<Range*>( index->asInst() );

      {
         InstanceLock::Locker( &m_lock, &str );

         int64 strLen = str.length();
         int64 start = rng.start();
         int64 end = ( rng.isOpen() ) ? strLen : rng.end();

         // handle negative indexes
         if ( start < 0 ) start = strLen + start;
         if ( end < 0 ) end = strLen + end;

         // do some validation checks before proceeding
         if ( start >= strLen  || end > strLen )
         {
            throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("index out of range") );
         }

         if ( value->isString() )  // should be a string
         {
            String& strVal = *value->asString();
            str.change( (Falcon::length_t)start, (Falcon::length_t)end, strVal );
         }
         else
         {
            String temp;
            temp.append((char_t)value->forceInteger());
            str.change((Falcon::length_t)start, (Falcon::length_t)end, temp );
         }
      }

      ctx->stackResult( 3, *value );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("invalid assignment to string") );
   }
}


//=======================================================================
// Comparation
//

void ClassString::op_compare( VMContext* ctx, void* self ) const
{
   Item* op1, *op2;

   OpToken token( ctx, op1, op2 );

   String* string = static_cast<String*>( self );

   Class* otherClass;
   void* otherData;

   if ( op2->asClassInst( otherClass, otherData ) )
   {
      if ( otherClass->typeID() == typeID() )
      {
         token.exit( string->compare(*static_cast<String*>( otherData ) ) );
      }
      else
      {
         token.exit( typeID() - otherClass->typeID() );
      }
   }
   else
   {
      token.exit( typeID() - op2->type() );
   }
}


void ClassString::op_toString( VMContext* ctx, void* data ) const
{
   // this op is generally called for temporary items,
   // ... so, even if we shouldn't be marked,
   // ... we won't be marked long if we're temporary.
   ctx->topData().setUser( this, data );
}


void ClassString::op_isTrue( VMContext* ctx, void* str ) const
{
   /* No lock -- we can accept sub-program level uncertainty */
   ctx->topData().setBoolean( static_cast<String*>( str )->size() != 0 );
}


void ClassString::op_iter( VMContext* ctx, void* self ) const
{
   /* No lock -- we can accept sub-program level uncertainty */
   length_t size = static_cast<String*>( self )->size();
   if( size == 0 ) {
      ctx->pushData(Item()); // we should not loop
   }
   else
   {
      ctx->pushData(Item(0));
   }
}


void ClassString::op_next( VMContext* ctx, void* instance ) const
{
   length_t pos = (length_t) ctx->topData().asInteger();

   String* self = static_cast<String*>(instance);
   InstanceLock::Token* tk = m_lock.lock(self);
   char_t chr = self->getCharAt(pos);
   ++pos;
   bool isLast = self->length() <= pos;
   m_lock.unlock(tk);

   ctx->topData().setInteger(pos);
   String* schr = new String;
   schr->append(chr);
   ctx->pushData( FALCON_GC_HANDLE(schr));
   if( ! isLast ) ctx->topData().setDoubt();
}


}

/* end of classstring.cpp */
