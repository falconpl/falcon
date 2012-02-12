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

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

namespace Falcon {

namespace StringProperties {
//
// Class properties used for enumeration
//
const int NUM_OF_PROPERTIES   = 26;

char properties[ ][ 12 ] = {
   "back",  // 1
   "charSize",
   "cmpi",
   "endsWith",
   "escape",
   "esq",
   "fill",
   "find",
   "ftrim",
   "join",  // 10
   "lower",
   "merge",
   "ptr",
   "replace",
   "replicate",
   "rfind",
   "rtrim",
   "split",
   "splittr",
   "startsWith",  // 20
   "toMemBuf",
   "trim",
   "unescape",
   "unesq",
   "upper",
   "wmatch" // 26
};

}


ClassString::ClassString():
   Class( "String", FLC_CLASS_ID_STRING )
{
}


ClassString::~ClassString()
{
}


void ClassString::dispose( void* self ) const
{
   delete static_cast<String*>( self );
}


void* ClassString::clone( void* source ) const
{
   return new String( *( static_cast<String*>( source ) ) );
}

void* ClassString::createInstance() const
{
   return new String;
}

void ClassString::store( VMContext*, DataWriter* dw, void* data ) const
{
   dw->write( *( static_cast<String*>( data ) ) );
}


void ClassString::restore( VMContext* , DataReader* dr, void*& data ) const
{
   String* str = new String;

   try
   {
      dr->read( *str );
      data = str;
   }
   catch( ... )
   {
      delete str;
      throw;
   }
}


void ClassString::describe( void* instance, String& target, int, int maxlen ) const
{
   String* self = static_cast<String*>( instance );

   target.size( 0 );
   target.append( '"' );

   // if ( (int) self->length() > maxlen )
   if ( (int) ( static_cast<String*>( instance ) )->length() > maxlen )
   {
      target.append( self->subString( 0, maxlen ) );
      target.append( "..." );
   }
   else
   {
      target.append( *self );
   }

   target.append( '"' );
}


void ClassString::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{
   for ( int cnt = 0; cnt < ( StringProperties::NUM_OF_PROPERTIES - 1 ); cnt++ )
   {
      cb( StringProperties::properties[ cnt ], false );
   }

   cb( StringProperties::properties[ StringProperties::NUM_OF_PROPERTIES - 1 ], true );
}


void ClassString::enumeratePV( void* self, Class::PVEnumerator& cb ) const
{
   // static_cast to string then get length
   Item temp = ( (int64)( static_cast<String*>( self ) )->length() );

   cb( "len_", temp );
}


bool ClassString::hasProperty( void*, const String& prop ) const
{
   for ( int cnt = 0; cnt < StringProperties::NUM_OF_PROPERTIES; cnt++ )
   {
      if ( prop == StringProperties::properties[ cnt ] )
      {
         return true;
      }
   }

   return false;
}


void ClassString::gcMark( void* instance, uint32 mark ) const
{
   static_cast<String*>( instance )->gcMark( mark );
}


bool ClassString::gcCheck( void* instance, uint32 mark ) const
{
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
      String* copy = new String( *str );

      copy->append( op2->describe() );

      ctx->stackResult( 2, copy->garbage() );

      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      String *copy = new String( *str );

      copy->append( *static_cast<String*>( inst ) );

      ctx->stackResult( 2, copy->garbage() );

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

      deep->prepend( *str );
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
         long depth = ctx->codeDepth();
         
         Class* cls;
         void* data;                 
         itm->forceClassInst( cls, data )
         cls->op_toString( ctx, data );
         
         if( depth == ctx->codeDepth() )
         {
            ctx->popCode();
            ctx->popData();
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

   Class* cls;
   void* inst;

   if ( ! op2->asClassInst( cls, inst ) )
   {
      if ( op1->copied() )
      {
         String* copy = new String( *str );

         copy->append( op2->describe() );

         ctx->stackResult( 2, copy->garbage() );
      }
      else
      {
         op1->asString()->append( op2->describe() );
      }

      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      if ( op1->copied() )
      {
         String *copy = new String( *static_cast<String*>( inst ) );

         copy->append( *str );

         ctx->stackResult( 2, copy->garbage() );
      }
      else
      {
         op1->asString()->append( *static_cast<String*>( inst ) );
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

      // op2 has been transformed
      String* deep = (String*) op2->asInst();

      deep->prepend( *str );
   }
}


ClassString::NextOp::NextOp()
{
   apply = apply_;
}


void ClassString::NextOp::apply_( const PStep*, VMContext* ctx )
{
   // The result of a deep call is in A
   Item* op1, *op2;

   ctx->operands( op1, op2 ); // we'll discard op2

   String* deep = op2->asString();
   String* self = op1->asString();

   if( op1->copied() )
   {
      String* copy = new String( *self );
      copy->append( *deep );
      ctx->stackResult( 2, copy->garbage() );
   }
   else
   {
      ctx->popData();
      self->append( *deep );
   }
   
   ctx->popCode();
}


ClassString::InitNext::InitNext()
{
   apply = apply_;
}


void ClassString::InitNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->opcodeParam(1)->asString()->copy( *ctx->topData().asString() );
   ctx->popData();
   ctx->popCode();
}

void ClassString::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *stritem;

   ctx->operands( stritem, index );

   String& str = *static_cast<String*>( self );

   if ( index->isOrdinal() )
   {
      int64 v = index->forceInteger();

      if ( v < 0 ) v = str.length() + v;

      if ( v >= str.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
      }

      String *s = new String();

      s->append( str.getCharAt( v ) );

      ctx->stackResult( 2, s->garbage() );
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
      if ( ! cls->typeID() == FLC_CLASS_ID_RANGE )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "unknown index" ) );
      }

      Range& rng = *static_cast<Range*>( udata );

      int64 step = ( rng.step() == 0 ) ? 1 : rng.step(); // assume 1 if no step given
      int64 start = rng.start();
      int64 end = rng.end();
      int64 strLen = str.length();

      // do some validation checks before proceeding
      if ( start >= strLen || start < ( strLen * -1 )  || end > strLen || end < ( strLen * -1 ) )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
      }

      bool reverse = false;

      String *s = new String();

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
            s->append( str.getCharAt( start ) );
            start += step;
         }
      }
      else
      {
         while ( start < end )
         {
            s->append( str.getCharAt( start ) );
            start += step;
         }
      }

      ctx->stackResult( 2, s->garbage() );
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

   if ( ! value->isString() )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("non string/char being assigned") );
   }

   String& strData = *( value->asString() );

   if ( index->isOrdinal() )
   {
      // simple index assignment: a[x] = value

      int64 v = index->forceInteger();

      if ( v < 0 ) v = str.length() + v;

      if ( v >= str.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("index out of range") );
      }

      value->copied( true ); // the value is copied here.

      str.setCharAt( v, strData.getCharAt( 0 ) );

      ctx->stackResult( 3, *value );
   }
   else if ( ! index->isUser() )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("invalid assignment to string") );
   }

   Class *rangeClass, *rhs;
   void *udataRangeInst, *udataRhs;

   if ( ! index->asClassInst( rangeClass, udataRangeInst ) )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("lhs index error") );
   }

   if ( rangeClass->typeID() != FLC_CLASS_ID_RANGE )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("unknown index range") );
   }

   if ( ! value->asClassInst( rhs, udataRhs ) ) // Get the rhs class and instance
   {
      // Something is not right
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ) );
   }

   Range& rng = *static_cast<Range*>( udataRangeInst );

   int64 strLen = str.length();
   int64 start = rng.start();
   int64 end = ( rng.isOpen() ) ? strLen : rng.end();

   // handle negative indexes
   if ( start < 0 ) start = strLen + start;
   if ( end < 0 ) end = strLen + end;

   int64 rangeLen = ( rng.isOpen() ) ? strLen - start : end - start;

   // do some validation checks before proceeding
   if ( start >= strLen || start < ( strLen * -1 )  || end > strLen || end < ( strLen * -1 ) )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("index out of range") );
   }

   if ( rhs->typeID() == FLC_CLASS_ID_STRING )  // should be a string
   {
      String& strVal = *static_cast<String*>( udataRhs );

      int64 rhsLen = strVal.length();

      if ( rangeLen < rhsLen )
      {
         str.change( (Falcon::length_t)start, (Falcon::length_t)end, strVal );
      }
      else if ( rangeLen > rhsLen )
      {
         str.change( (Falcon::length_t)start, (Falcon::length_t)end, strData );
      }
      else // rangeLen == rhsLen
      {
         str.copy( strData );
      }

      value->copied( true ); // the value is copied here.

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
   ctx->topData().setBoolean( static_cast<String*>( str )->size() != 0 );
}

}

/* end of classstring.cpp */
