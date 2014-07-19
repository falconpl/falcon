/*
   FALCON - The Falcon Programming Language.
   FILE: tokenizer.cpp

   String tokenizer general purpose class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Apr 2014 15:01:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/cm/tokenizer.cpp"

#include <falcon/stringtok.h>
#include <falcon/stderrors.h>
#include <falcon/cm/tokenizer.h>
#include <falcon/function.h>
#include <falcon/item.h>
#include <falcon/vmcontext.h>

namespace Falcon {
namespace {

/*#
 @class Tokenizer
 @brief Helper for iterative and generator-based sub-string extractor.
 @param string The string to be tokenized
 @param token The token used for tokenization
 @optparam group True to group sequences on tokens
 @optparam limit Count of maximum number of tokens

 This class performs an iterative tokenization over a given string, using a single element
 as separator.

 This class has support for generator usage (can be used in for/in loops, advance() function,
 ^[] iterator operator, forEach() BOM method etc.), and it provides next()/hasNext() methods
 that can be used directly to extract the desired tokens.

 This class is meant to efficiently break a string in sub-parts using a simple
 single-token tokenization. The @a MultiTokenizer provides a less efficient but
 more flexible support that allows to break a string using multiple tokens and regular
 expression-based tokens.
*/
FALCON_DECLARE_FUNCTION( init, "string:S,token:[S],group:[B],limit:[N]" );
FALCON_DEFINE_FUNCTION_P1( init )
{
   Item* i_string = ctx->param(0);
   Item* i_token = ctx->param(1);
   Item* i_group = ctx->param(2);
   Item* i_limit = ctx->param(3);

   String* string;
   String* token;
   bool group;
   int64 limit;

   String dflt(" ");
   dflt.bufferize();
   bool bCheck = FALCON_PCHECK_GET( i_string, String, string )
              && FALCON_PCHECK_O_GET( i_token, String, token, &dflt )
              && FALCON_PCHECK_O_GET( i_group, Boolean, group, false )
              && FALCON_PCHECK_O_GET( i_limit, Ordinal, limit, -1 );

   if( ! bCheck )
   {
      throw paramError();
   }


   StringTokenizer* tk = new StringTokenizer(*string, *token, group );
   if( limit >= 0 )
   {
      tk->setLimit((uint32) limit);
   }

   ctx->self() = FALCON_GC_STORE(methodOf(), tk);
   ctx->returnFrame(ctx->self());
}

/*#
 @method next Tokenizer
 @return the next tokenized substring or nil of none.

 */
FALCON_DECLARE_FUNCTION( next, "" );
FALCON_DEFINE_FUNCTION_P1( next )
{
   StringTokenizer* tk = ctx->tself<StringTokenizer>();
   const Ext::ClassTokenizer* cls = static_cast<const Ext::ClassTokenizer*>(methodOf());

   if( tk->hasNext() )
   {
      String *tgt = new String;
      InstanceLock::Token* lk = cls->m_lock.lock(tk);
      tk->next(*tgt);
      cls->m_lock.unlock(lk);
      ctx->returnFrame(FALCON_GC_HANDLE(tgt));
   }
   else
   {
      ctx->returnFrame();
   }
}

/*#
 @property hasNext Tokenizer
 @brief This property is true when @a Tokenizer.next can return more elements.
*/
static void get_hasNext(const Class* ocls, const String&, void* data, Item& value )
{
   StringTokenizer* tk = static_cast<StringTokenizer*>(data);
   const Ext::ClassTokenizer* cls = static_cast<const Ext::ClassTokenizer*>(ocls);


   InstanceLock::Token* lk = cls->m_lock.lock(tk);
   value.setBoolean( tk->hasNext() );
   cls->m_lock.unlock(lk);
}

}


namespace Ext {

ClassTokenizer::ClassTokenizer():
         Class("Tokenizer")
{
   setConstuctor( new FALCON_FUNCTION_NAME(init) );
   addMethod( new FALCON_FUNCTION_NAME(next) );

   addProperty("hasNext", &get_hasNext);
}


ClassTokenizer::~ClassTokenizer()
{
}

void ClassTokenizer::dispose( void* instance ) const
{
   StringTokenizer* st = static_cast<StringTokenizer*>(instance);
   delete st;
}

void* ClassTokenizer::clone( void* instance ) const
{
   StringTokenizer* st = static_cast<StringTokenizer*>(instance);
   return new StringTokenizer(*st);
}

void ClassTokenizer::gcMarkInstance( void* instance, uint32 mark ) const
{
   StringTokenizer* st = static_cast<StringTokenizer*>(instance);
   st->gcMark(mark);
}

bool ClassTokenizer::gcCheckInstance( void* instance, uint32 mark ) const
{
   StringTokenizer* st = static_cast<StringTokenizer*>(instance);
   return st->currentMark() >= mark;
}


void* ClassTokenizer::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

//==================================================================
// Operators
//

void ClassTokenizer::op_iter( VMContext* ctx, void* self ) const
{
   StringTokenizer* st = static_cast<StringTokenizer*>(self);
   /* No lock -- we can accept sub-program level uncertainty */

   if( ! st->hasNext() ) {
      ctx->pushData(Item()); // we should not loop
   }
   else
   {
      ctx->pushData(FALCON_GC_STORE(this, new StringTokenizer(*st)));
   }
}


void ClassTokenizer::op_next( VMContext* ctx, void* ) const
{
   fassert( ctx->topData().asClass() == this );
   StringTokenizer* st = static_cast<StringTokenizer*>(ctx->topData().asInst());
   String* tgt = new String;

   InstanceLock::Token* tk = m_lock.lock(st);
   if( st->next(*tgt) )
   {
      m_lock.unlock(tk);
      ctx->pushData( FALCON_GC_HANDLE(tgt) );
      if( st->hasNext() )
      {
         ctx->topData().setDoubt();
      }
   }
   else {
      m_lock.unlock(tk);
      delete tgt;
      ctx->pushBreak();
   }
}
}

}

/* end of tokenizer.cpp */
