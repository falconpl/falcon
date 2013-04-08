/*
   FALCON - The Falcon Programming Language.
   FILE: classmstring.cpp

   Mutable String type handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Apr 2013 15:21:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classmstring.cpp"

#include <falcon/classes/classmstring.h>
#include <falcon/range.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/errors/accesserror.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/paramerror.h>

#include <falcon/function.h>

namespace Falcon {

//=====================================================================
// Properties
//
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

static void get_charSize( const Class*, const String&, void* instance, Item& value )
{
   String* str = static_cast<String*>(instance);
   value.setInteger( str->manipulator()->charSize() );
}

static void set_charSize( const Class*, const String&, void* instance, const Item& value )
{
   String* str = static_cast<String*>(instance);

   if( ! value.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_inv_params, __LINE__, SRC )
         .extra( "N" ) );
   }

   uint32 bpc = (uint32) value.isOrdinal();
   if ( ! str->setCharSize( bpc ) )
   {
      throw new  OperandError( ErrorParam( e_param_range, __LINE__, SRC ) );
   }
}



namespace _classMString
{
/*#
   @method fill String
   @brief Fills a string with a given character or substring.
   @param chr The character (unicode value) or substring used to refill this string.
   @return The string itself.

   This method fills the physical storage of the given string with a single
   character or a repeated substring. This can be useful to clean a string used repeatedly
   as input buffer.

   @note When used statically as a class method, the first parameter can be a mutable string.
*/

FALCON_DECLARE_FUNCTION( fill, "target:MString,chr:N|S" );

FALCON_DEFINE_FUNCTION_P1(fill)
{
   Item *i_string;
   Item *i_chr;

   // Parameter checking;
   if ( ctx->isMethodic() )
   {
      i_string = &ctx->self();
      i_chr = ctx->param(0);
   }
   else
   {
      i_string = ctx->param(0);
      i_chr = ctx->param(1);
   }

   if( i_string == 0 || ! i_string->asClass()->isDerivedFrom( methodOf() )
      || i_chr == 0 || ( ! i_chr->isOrdinal() && !i_chr->isString())
      )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic() );
   }

   String *string = i_string->asString();

   if ( i_chr->isOrdinal() )
   {
      uint32 chr = (uint32) i_chr->forceInteger();
      for( uint32 i = 0; i < string->length(); i ++ )
      {
         string->setCharAt( i, chr );
      }
   }
   else
   {
      String* rep = i_chr->asString();

      if ( rep->length() == 0 )
      {
          throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .extra( "Empty fill character" ) );
      }

      uint32 pos = 0;
      uint32 pos2 = 0;
      while( pos < string->length() )
      {
         string->setCharAt( pos++, rep->getCharAt( pos2++ ) );
         if ( pos2 >= rep->length() )
         {
            pos2 = 0;
         }
      }
   }

   ctx->returnFrame( Item(methodOf(), string) );
}
}

//
// Class properties used for enumeration
//

ClassMString::ClassMString():
   ClassString( "MString" )
{
   addProperty( "isText", &get_isText, &set_isText );
   addProperty( "charSize", &get_charSize, &set_charSize );

   addMethod( new _classMString::Function_fill, true );
}

ClassMString::~ClassMString()
{
}


void ClassMString::describe( void* instance, String& target, int depth, int maxlen ) const
{
   ClassString::describe(instance, target, depth, maxlen);
   target = "m" +target;
}


void ClassMString::op_aadd( VMContext* ctx, void* self ) const
{
   String* str = static_cast<String*>( self );

   Item* op1, *op2;
   ctx->operands( op1, op2 );

   Class* cls=0;
   void* inst=0;

   if ( op2->isString() )
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
      return;
   }
   else if ( ! op2->asClassInst( cls, inst ) )
   {
      InstanceLock::Token* tk = m_lock.lock(op1->asString());
      op1->asString()->append( op2->describe() );
      m_lock.unlock(tk);
      ctx->popData();
      return;
   }


   // else we surrender, and we let the virtual system to find a way.
   ctx->pushCode( m_nextOp );
   long depth = ctx->codeDepth();

   // this will transform op2 slot into its string representation.
   cls->op_toString( ctx, inst );

   if( ctx->codeDepth() == depth )
   {
      ctx->popCode();

      // op2 has been transformed (and is ours)
      String* deep = (String*) op2->asInst();

      InstanceLock::Token* tk = m_lock.lock(str);
      deep->prepend( *str );
      m_lock.unlock(tk);
   }
}

void ClassMString::op_amul( VMContext* ctx, void* instance ) const
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
   target = self;

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


void ClassMString::op_adiv( VMContext* ctx, void* instance ) const
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
   self->append((char_t) count);
   m_lock.unlock(tk);
}


void ClassMString::op_setIndex( VMContext* ctx, void* self ) const
{
   Item* value, *arritem, *index;

   ctx->operands( value, arritem, index );

   String& str = *static_cast<String*>( self );

   if ( ! value->isString() && ! value->isOrdinal())
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "S" ) );
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
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "I|R" ) );
   }
}

}

/* end of classstring.cpp */
