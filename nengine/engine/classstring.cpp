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

#include <falcon/classstring.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/accesserror.h>

namespace Falcon {

ClassString::ClassString():
   Class("String", FLC_CLASS_ID_STRING )
{
}


ClassString::~ClassString()
{
}


void ClassString::dispose( void* self ) const
{
   String* f = static_cast<String*>(self);
   delete f;
}


void* ClassString::clone( void* source ) const
{
   String* s = static_cast<String*>(source);
   return new String(*s);
}


void ClassString::serialize( DataWriter* stream, void* self ) const
{
   String* s = static_cast<String*>(self);
   s->serialize(stream);
}


void* ClassString::deserialize( DataReader* stream ) const
{
   String* s = new String;
   try {
      s->deserialize( stream );
   }
   catch( ... )
   {
      delete s;
      throw;
   }

   return s;
}

void ClassString::describe( void* instance, String& target, int, int maxlen ) const
{
   String* self = static_cast<String*>(instance);
   target.size(0);
   target.append('"');
   if( (int) self->length() > maxlen )
   {
      target += self->subString(0, maxlen);
      target += "...";
   }
   else
   {
      target += *self;
   }
   
   target.append('"');
}

void ClassString::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{
   // TODO: More
   cb("len", false );
   cb("len_", false );
   cb("__add", true );
}

void ClassString::enumeratePV( void* self, Class::PVEnumerator& cb ) const
{
   // TODO: More
   String* str = static_cast<String*>(self); 
   Item temp;
   
   temp = ((int64)str->length());
   cb("len_", temp );
}

bool ClassString::hasProperty( void*, const String& prop ) const
{
   // TODO: More
   return 
      prop == "len"
      || prop == "len_"
      || prop == "__add";
}

//=======================================================================
// Addition

void ClassString::op_add( VMContext* ctx, void* self ) const
{
   String* str = static_cast<String*>(self);
   Item* op1, *op2;
   ctx->operands(op1, op2);

   Class* cls;
   void* inst;
   if( ! op2->asClassInst( cls, inst ) )
   {
      String* copy = new String(*str);
      copy->append(op2->describe());
      ctx->stackResult(2, copy->garbage() );
      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      String *copy = new String(*str);
      copy->append( *static_cast<String*>(inst) );
      ctx->stackResult(2, copy->garbage() );
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
      String* deep = (String*)op2->asInst();
      deep->prepend( *str );
   }
}

//=======================================================================
// Auto Addition
//

void ClassString::op_create( VMContext* ctx, int pcount ) const
{
   // no param?
   if( pcount == 0 )
   {
      // create a string.
      String* s = new String;
      ctx->stackResult(1, s->garbage() );
   }
   else
   {
      // the parameter is a string?
      Item* itm = ctx->opcodeParams(pcount);
      if( itm->isString() )
      {
         // copy it.
         String* s = new String( *itm->asString() );
         ctx->stackResult( pcount + 1, s->garbage() );
      }
      else
      {
         // apply the op_toString on the item.
         Item cpy = *itm;
         ctx->stackResult( pcount + 1, cpy );

         Class* cls;
         void* data;
         cpy.forceClassInst( cls, data );
         cls->op_toString( ctx, data );
      }
   }
}


void ClassString::op_aadd( VMContext* ctx, void* self ) const
{
   String* str = static_cast<String*>(self);
   Item* op1, *op2;
   ctx->operands(op1, op2);

   Class* cls;
   void* inst;
   if( ! op2->asClassInst( cls, inst ) )
   {
      if( op1->copied() )
      {
         String* copy = new String(*str);
         copy->append(op2->describe());
         ctx->stackResult(2, copy->garbage() );
      }
      else
      {
         op1->asString()->append(op2->describe());
      }

      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      if( op1->copied() )
      {
         String *copy = new String(*str);
         copy->append( *static_cast<String*>(inst) );
         ctx->stackResult(2, copy->garbage() );
      }
      else
      {
         op1->asString()->append( *static_cast<String*>(inst) );
      }
      return;
   }

   // else we surrender, and we let the virtual system to find a way.
   ctx->pushCode( &m_nextOp );

   // this will transform op2 slot into its string representation.
   cls->op_toString( ctx, inst );

   if( ! ctx->wentDeep(&m_nextOp) )
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
   ctx->operands(op1, op2); // we'll discard op2

   String* deep = ctx->regA().asString();
   String* self = op1->asString();

   if( op1->copied() )
   {
      String* copy = new String(*self);
      copy->append( *deep );
      ctx->stackResult( 2, copy->garbage() );
   }
   else
   {
      ctx->popData();
      self->append(*deep);
   }
}

void ClassString::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *stritem;
   ctx->operands( stritem, index );

   String& str = *static_cast<String*>(self);
   if (index->isOrdinal())
   {
      int64 v = index->forceInteger();
      if( v < 0 ) v = str.length() + v;
      if( v >= str.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("out of range") );
      }
      String *s = new String();
      s->append(str.getCharAt(v));
      ctx->stackResult(2, s->garbage() );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("N") );
   }
}

//=======================================================================
// Comparation
//

void ClassString::op_compare( VMContext* ctx, void* self ) const
{
   Item* op1, *op2;
   OpToken token( ctx, op1, op2 );
   String* string = static_cast<String*>(self);

   Class* otherClass;
   void* otherData;

   if( op2->asClassInst( otherClass, otherData ) )
   {
      if( otherClass->typeID() == typeID() )
      {
         token.exit( string->compare(*static_cast<String*>(otherData) ) );
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


void ClassString::op_toString( VMContext*, void* ) const
{
   // nothing to do -- the topmost item of the stack is already a string.
}

void ClassString::op_true( VMContext* ctx, void* str) const
{
   ctx->topData() = static_cast<String*>(str)->size() != 0;
}

}

/* end of classstring.cpp */
