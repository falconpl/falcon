/*
   FALCON - The Falcon Programming Language.
   FILE: multitokenizer.cpp

   String tokenizer general purpose class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Apr 2014 15:01:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/cm/multitokenizer.cpp"

#include <falcon/streamtok.h>
#include <falcon/stdhandlers.h>
#include <falcon/stdsteps.h>
#include <falcon/stderrors.h>
#include <falcon/cm/multitokenizer.h>
#include <falcon/function.h>
#include <falcon/item.h>
#include <falcon/textreader.h>
#include <falcon/vmcontext.h>
#include <falcon/processor.h>
#include <falcon/gclock.h>

namespace Falcon {
namespace {

class MultiTokenizer: public StreamTokenizer
{
public:
   MultiTokenizer(const String& src ):
      StreamTokenizer(src)
   {
      setOwnText(false);
      setOwnToken(false);
   }


   MultiTokenizer( TextReader* tr, uint32 bufsize = StreamTokenizer::DEFAULT_BUFFER_SIZE ):
      StreamTokenizer(tr, bufsize)
   {
      setOwnText(false);
      setOwnToken(false);
   }

   MultiTokenizer( const MultiTokenizer& other ):
      StreamTokenizer(other)
   {}


   virtual ~MultiTokenizer() {}


   virtual void onTokenFound( int32 id, String* textContent, String* tokenContent, void* userData )
   {
      static const PStep* rt = &Engine::instance()->stdSteps()->m_returnFrameWithTop;
      static const PStep* rtd = &Engine::instance()->stdSteps()->m_returnFrameWithTopDoubt;

      VMContext* ctx = Processor::currentProcessor()->currentContext();
      GCLock* glk = static_cast<GCLock*>(userData);

      if( glk == 0 )
      {
         delete tokenContent;
         if( hasNext() )
         {
            ctx->returnFrameDoubt(FALCON_GC_HANDLE(textContent));
         }
         else
         {
            ctx->returnFrame(FALCON_GC_HANDLE(textContent));
         }
      }
      else {
         const Item& cb = glk->item();
         ctx->pushCode( hasNext() ? rtd : rt );
         int count;
         Item params[3];

         if( tokenContent != 0 )
         {
            params[0] = (int64) id;
            params[1] = FALCON_GC_HANDLE(textContent);
            params[2] = FALCON_GC_HANDLE(tokenContent);
            count = 3;
         }
         else {
            count = 1;
            params[0] = FALCON_GC_HANDLE(textContent);
         }

         ctx->callItem(cb, count, params);
      }
   }
};

/*#
 @class MultiTokenizer
 @brief Advanced helper for iterative and generator-based sub-string extractor.
 @param source The string to be tokenized or a @a Stream
 @optparam enc Text-encoding of the given stream (defaults to "utf8").

 This class performs an iterative tokenization over a given string, using multiple
 tokens and providing information about which token is actually found.

 The tokens can be simple text-based tokens or whole regular expressions,
 are added separately using the @a MultiTokenizer.addToken() or @a MultiTokenizer.addRE()
 methods.

 Do not use precompiled regular expressions (r"" strings), or instances of the @a RE class to
 add a regular expression as a token in an instance of this class; the instance will create
 its own copy of the regular expression out of the addRE() method parameter.

 This class has support for generator usage (can be used in for/in loops, advance() function,
 ^[] iterator operator, forEach() BOM method etc.), and it provides next()/hasNext() methods
 that can be used directly to extract the desired tokens.


*/
FALCON_DECLARE_FUNCTION( init, "source:S|Stream,enc:[S]" );
FALCON_DEFINE_FUNCTION_P1( init )
{
   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   Item* i_source= ctx->param(0);
   Item* i_enc = ctx->param(1);

   if( i_source == 0 || ! (i_source->isString() || i_source->isInstanceOf(clsStream))
       || (i_enc != 0 && ! i_enc->isString() )
       )
   {
      throw paramError();
   }

   String dflt("utf8");
   String* enc;
   if( i_enc == 0 )
   {
      enc = &dflt;
   }
   else {
      enc = i_enc->asString();
   }

   MultiTokenizer* tk;
   if( i_source->isString() )
   {
      tk = new MultiTokenizer(*i_source->asString());
   }
   else {
      Transcoder* tc = Engine::instance()->getTranscoder(*enc);
      if( tc == 0 )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("Invalid encoding "+*enc));
      }

      Stream* stream = i_source->castInst<Stream>(clsStream);
      TextReader* tr = new TextReader( stream, tc );
      tk = new MultiTokenizer(tr);
      tr->decref();
   }

   ctx->self() = FALCON_GC_STORE(methodOf(), tk);
   ctx->returnFrame(ctx->self());
}

/*#
 @method next MultiTokenizer
 @return the next tokenized substring or nil of none.

 */
FALCON_DECLARE_FUNCTION( next, "" );
FALCON_DEFINE_FUNCTION_P1( next )
{
   MultiTokenizer* tk = ctx->tself<MultiTokenizer*>();

   if( ! tk->next() )
   {
      ctx->returnFrame();
   }
}

/*#
 @property hasNext MultiTokenizer
 @brief This property is true when @a Tokenizer.next can return more elements.
*/
static void get_hasNext(const Class*, const String&, void* data, Item& value )
{
   MultiTokenizer* tk = static_cast<MultiTokenizer*>(data);
   value.setBoolean( tk->hasNext() );
}


// Deletor used to free up the callbacks used by the multi-tokenizer.
static void internal_unlocker( void* unlk )
{
   GCLock* glk = static_cast<GCLock*>(unlk);
   glk->dispose();
}


static void internal_add( Function* caller, VMContext* ctx, bool isRE )
{
   Item* i_token = ctx->param(0);
   Item* i_cb = ctx->param(1);
   String* token;

   bool bCheck = FALCON_PCHECK_GET( i_token, String, token ) && (i_cb == 0 || i_cb->isCallable() );
   if( ! bCheck ) {
      throw caller->paramError();
   }

   MultiTokenizer* tk = ctx->tself<MultiTokenizer*>();
   if (isRE)
   {
      tk->addRE(*token, i_cb == 0 ? 0 : Engine::instance()->GC_lock(*i_cb), &internal_unlocker );
   }
   else {
      tk->addToken(*token, i_cb == 0 ? 0 : Engine::instance()->GC_lock(*i_cb), &internal_unlocker );
   }

   ctx->returnFrame(ctx->self());
}

/*#
 @method addToken MultiTokenizer
 @brief Adds a token to the multi tokenizer.
 @param token The token to be added
 @optparam cb A callback code that will be evaluated if the token is found.
 @return This instance

 If the @b cb callback is not provided, when this token is found, the @b next()
 method returns the string between the last position and the beginning of the found
 token.

 If a callback is provided, then it gets called before @b next return with the
 following parameters:
 # the ordinal ID of the token.
 # the found string.
 # the token itself.

 The residual string (the one between the last found token and the end of the stream)
 can be processed using the method @b onResidual.

 The @b next() call will return the return value of the the callback.

 The following is a working example:

 @code
 mt = MultiTokenizer( "Hello World" )
 mt.addToken( " ", { id, string, token =>
          > "Found word: '", string, "'"
          if id >= 0: > "The token was '",token,"'"
          return string
          } )
  mt.onResidual( {t => > "Last part: '", t, "'"; return t } )

 ^[ mt ]   // equivalent to while mt.next(); end

 @endcode
 */
FALCON_DECLARE_FUNCTION( addToken, "token:S,cb:[C]" );
FALCON_DEFINE_FUNCTION_P1( addToken )
{
   internal_add(this, ctx, false);
}

/*#
 @method addRE MultiTokenizer
 @brief Adds a regular expression to the multi tokenizer.
 @param re A string that will be interpreted as the regular expression.
 @optparam cb A callback code that will be evaluated if the expression is found.
 @return This instance


 This method works analogously to @a MultiTokenizer.addToken, the only difference being
 the fact that the string passed will be interpreted not as a literal token, but as a
 regular expression.

 If @b cb is provided, when the regular expression is found it's @b token parameter
 will contain the expansion of the found regular expression. Eventual parenthetized sub-regular
 expressions are discarded.

 The following is a working example:

 @code
 mt = MultiTokenizer( "Hello World" );
 mt.addRE( "[A-Z]| ", { id, string, token =>
          > "Found word: '", string, "'"
          if id >= 0: > "The token was '",token,"'"
          return string
          } )
 mt.onResidual( {t => > "Last part: '", t, "'"; return t } )

 ^[ mt ]   // equivalent to while mt.next(); end

 @endcode
 */
FALCON_DECLARE_FUNCTION( addRE, "token:S,cb:[C]" );
FALCON_DEFINE_FUNCTION_P1( addRE )
{
   internal_add( this, ctx, true );
}

FALCON_DECLARE_FUNCTION( onResidual, "cb:[C]" );
FALCON_DEFINE_FUNCTION_P1( onResidual )
{
   Item* i_cb = ctx->param(0);

   if( i_cb == 0 || ! i_cb->isCallable() )
   {
      throw paramError();
   }

   MultiTokenizer* tk = ctx->tself<MultiTokenizer*>();
   tk->onResidual(Engine::instance()->GC_lock(*i_cb), &internal_unlocker);
   ctx->returnFrame(ctx->self());
}


}


namespace Ext {

ClassMultiTokenizer::ClassMultiTokenizer():
         Class("MultiTokenizer")
{
   m_mthNext = new FALCON_FUNCTION_NAME(next);

   setConstuctor( new FALCON_FUNCTION_NAME(init) );
   addMethod( m_mthNext );
   addMethod( new FALCON_FUNCTION_NAME(addToken) );
   addMethod( new FALCON_FUNCTION_NAME(addRE) );
   addMethod( new FALCON_FUNCTION_NAME(onResidual) );

   addProperty("hasNext", &get_hasNext);
}


ClassMultiTokenizer::~ClassMultiTokenizer()
{
}

void ClassMultiTokenizer::dispose( void* instance ) const
{
   MultiTokenizer* st = static_cast<MultiTokenizer*>(instance);
   delete st;
}

void* ClassMultiTokenizer::clone( void* instance ) const
{
   MultiTokenizer* st = static_cast<MultiTokenizer*>(instance);
   return st;
}

void ClassMultiTokenizer::gcMarkInstance( void* instance, uint32 mark ) const
{
   MultiTokenizer* st = static_cast<MultiTokenizer*>(instance);
   st->gcMark(mark);
}

bool ClassMultiTokenizer::gcCheckInstance( void* instance, uint32 mark ) const
{
   MultiTokenizer* st = static_cast<MultiTokenizer*>(instance);
   return st->currentMark() >= mark;
}


void* ClassMultiTokenizer::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

//==================================================================
// Operators
//

void ClassMultiTokenizer::op_iter( VMContext* ctx, void* self ) const
{
   MultiTokenizer* st = static_cast<MultiTokenizer*>(self);

   if( ! st->hasNext() ) {
      ctx->pushData(Item()); // we should not loop
   }
   else
   {
      ctx->pushData(Item(this, st));
   }
}


void ClassMultiTokenizer::op_next( VMContext* ctx, void* ) const
{
   fassert( ctx->topData().asClass() == this );

   Item self = ctx->topData();
   ctx->pushData(self);
   ctx->topData().methodize(m_mthNext);
   ctx->callInternal(m_mthNext, 0, self);
}

}
}

/* end of multitokenizer.cpp */
