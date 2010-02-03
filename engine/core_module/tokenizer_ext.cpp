/*
   FALCON - The Falcon Programming Language.
   FILE: tokenizer_ext.cpp

   Tokenizer class support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Feb 2009 22:58:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/tokenizer.h>


namespace Falcon {
namespace core {

/*#
   @class Tokenizer
   @brief Simple stream-oriented parser for efficient basic recognition of incoming data.
   @optparam seps A string representing the separators.
   @optparam options Tokenization options.
   @optparam tokLen Maximum length of returned tokens.
   @optparam source The string to be tokenized, or a stream to be read for tokens.

   The tokenizer class is meant to provide simple and efficient logic to parse incoming
   data (mainly, incoming from string).

   The source can also be set at a second time with the @a Tokenizer.parse method.
   @b seps defaults to " " if not given.

   The @b options parameter can be a binary combinations of the following values:

   - @b Tokenizer.groupsep: Groups different tokens into one. If not given, when a token immediately
        follows another, an empty field is returned.
   - @b Tokenizer.bindsep: Return separators inbound with their token.
   - @b Tokenizer.trim: trim whitespaces away from returned tokens.
   - @b Tokenizer.wsAsToken: Treat a sequence of whitespaces as a single token.
   - @b Tokenizer.retsep: Return separators as separate tokens.
*/

FALCON_FUNC  Tokenizer_init ( ::Falcon::VMachine *vm )
{
   Item* i_separators = vm->param(0);
   Item* i_options = vm->param(1);
   Item* i_len = vm->param(2);
   Item* i_source = vm->param(3);

   if ( ( i_separators != 0 && ! ( i_separators->isString() || i_separators->isNil()))
        || ( i_options != 0  && !( i_options->isInteger() || i_options->isNil() ) )
        || ( i_len != 0  && ! ( i_len->isNumeric() || i_len->isNil() ) )
        || ( i_source != 0 &&
              ( ! i_source->isString() &&
                 ( ! i_source->isObject() || ! i_source->asObjectSafe()->derivedFrom( "Stream" ) )))
   )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "[S],[I],[N],[S|Stream]" ) );
   }

   // we must have separators
   String seps;
   if( i_separators && i_separators->isString() )
      seps = *i_separators->asString();

   // Then, we may have options:
   TokenizerParams params;
   if( i_options != 0 && i_options->isInteger() )
   {
      uint32 opts = (uint32) i_options->asInteger();
      if( (opts & TOKENIZER_OPT_GRROUPSEP) != 0 )
         params.groupSep();
      if( (opts & TOKENIZER_OPT_BINDSEP) != 0 )
         params.bindSep();
      if( (opts & TOKENIZER_OPT_TRIM) != 0 )
         params.trim();
      if( (opts & TOKENIZER_OPT_RSEP) != 0 )
         params.returnSep();
      if( (opts & TOKENIZER_OPT_WSISTOK) != 0 )
         params.wsIsToken();
   }

   // ... and a maximum length?
   if ( i_len != 0 && i_len->isNumeric() )
   {
      params.maxToken( (int32) i_len->forceInteger() );
   }

   CoreObject* self = vm->self().asObject();

   // Finally determine the source.
   if ( i_source != 0 )
   {
      // save it from GC ripping
      self->setProperty( "_source", *i_source );

      if( i_source->isString() )
      {
         self->setUserData( new Tokenizer( params, seps, *i_source->asString() ) );
      }
      else
      {
         Stream* source = dyncast<Stream*>(i_source->asObject()->getFalconData() );
         self->setUserData( new Tokenizer( params, seps, source, false ) );
         if ( ! source->good() )
         {
            throw new IoError( ErrorParam( e_io_error, __LINE__ )
                  .origin( e_orig_runtime )
                  .sysError( (uint32) source->lastError() ) );
         }
      }
   }
   else
   {
      self->setUserData( new Tokenizer( params, seps ) );
   }
}

/*#
   @method parse Tokenizer
   @raise IoError on errors on the underlying stream.
   @brief Changes or set the source data for this tokenizer.
   @param source A string or a stream to be used as a source for the tokenizer.

   The first token is immediately read and set as the current token. If it's not
   empty, that is, if at least a token can be read, @a Tokenizer.hasCurrent
   returns true, and @a Tokenizer.token returns its value.
*/

FALCON_FUNC  Tokenizer_parse ( ::Falcon::VMachine *vm )
{
   Item* i_source = vm->param(0);

   if ( i_source == 0
         || ( ! i_source->isString() &&
            ( ! i_source->isObject() || ! i_source->asObjectSafe()->derivedFrom( "Stream" ) ))
   )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S|Stream" ) );
   }


   CoreObject* self = vm->self().asObject();
   self->setProperty( "_source", *i_source );
   Tokenizer* tzer = dyncast<Tokenizer*>( self->getFalconData() );

   if( i_source->isString() )
   {
      tzer->parse( *i_source->asString() );
   }
   else
   {
      Stream* source = dyncast<Stream*>(i_source->asObjectSafe()->getFalconData());
      tzer->parse( source, false );
      if ( ! source->good() )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .origin( e_orig_runtime )
               .sysError( (uint32) source->lastError() ) );
      }
   }
}

/*#
   @method rewind Tokenizer
   @brief Resets the status of the tokenizer.
   @raise IoError if the tokenizer is tokenizing a non-rewindable stream.
*/

FALCON_FUNC  Tokenizer_rewind ( ::Falcon::VMachine *vm )
{
   Tokenizer* tzer = dyncast<Tokenizer*>( vm->self().asObject()->getFalconData() );
   tzer->rewind();
}

/*#
   @method nextToken Tokenizer
   @brief Returns the next token from the tokenizer
   @return A string or nil at the end of the tokenization.
   @raise IoError on errors on the underlying stream.
   @raise CodeError if called on an unprepared Tokenizer.

   This method is actually a combination of @a Tokenizer.next followed by @a Tokenizer.token.

   Sample usage:
   @code
   t = Tokenizer( source|"A string to be tokenized" )
   while (token = t.nextToken()) != nil
      > "Token: ", token
   end
   @endcode

   @note When looping, remember to check the value of the returned token against nil,
         as empty strings can be legally returned multiple times, and they are considered false
         in logic checks.
*/

FALCON_FUNC  Tokenizer_nextToken ( ::Falcon::VMachine *vm )
{
   Tokenizer* tzer = dyncast<Tokenizer*>( vm->self().asObject()->getFalconData() );
   if( tzer->hasCurrent() )
   {
      CoreString* ret = new CoreString( tzer->getToken() );
      tzer->next();
      ret->bufferize();
      vm->retval( ret );
   }
   else
      vm->retnil();
}

/*#
   @method next Tokenizer
   @brief Advances the tokenizer up to the next token.
   @return True if a new token is now available, false otherwise.
   @raise IoError on errors on the underlying stream.
   @raise CodeError if called on an unprepared Tokenizer.


   For example:
   @code
   t = Tokenizer( source|"A string to be tokenized" )
   while t.hasCurrent()
      > "Token: ", t.token()
      t.next()
   end
   @endcode

   @see Tokenizer.token
*/

FALCON_FUNC  Tokenizer_next ( ::Falcon::VMachine *vm )
{
   Tokenizer* tzer = dyncast<Tokenizer*>( vm->self().asObject()->getFalconData() );
   vm->regA().setBoolean( tzer->next() );
}

/*#
   @method token Tokenizer
   @brief Get the current token.
   @return True if a new token is now available, false otherwise.
   @raise IoError on errors on the underlying stream.
   @raise CodeError if called on an unprepared Tokenizer, or before next().

   This method returns the current token.

   @see Tokenizer.nextToken
   @see Tokenizer.next
*/

FALCON_FUNC  Tokenizer_token ( ::Falcon::VMachine *vm )
{
   Tokenizer* tzer = dyncast<Tokenizer*>( vm->self().asObject()->getFalconData() );
   if( ! tzer->empty() )
   {
      CoreString* ret = new CoreString(  tzer->getToken() );
      ret->bufferize();
      vm->retval( ret );
   }
   else
      vm->retnil();
}

/*#
   @method hasCurrent Tokenizer
   @brief Return true if the tokenizer has a current token.
   @return True if a token is now available, false otherwise.

   Contrarily to iterators, it is necessary to call this @a Tokenizer.next
   at least once before calling this method.

   @see Tokenizer.nextToken
   @see Tokenizer.next
*/
FALCON_FUNC  Tokenizer_hasCurrent ( ::Falcon::VMachine *vm )
{
   Tokenizer* tzer = dyncast<Tokenizer*>( vm->self().asObject()->getFalconData() );
   vm->regA().setBoolean( tzer->hasCurrent() );
}

}
}

/* end of iterator_ext.cpp */
