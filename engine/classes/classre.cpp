/*
   FALCON - The Falcon Programming Language.
   FILE: classre.cpp

   RE2 object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Feb 2013 13:49:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classre.cpp"

#include <falcon/classes/classre.h>
#include <falcon/function.h>
#include <falcon/range.h>
#include <falcon/itemid.h>
#include <falcon/itemdict.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/range.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include "../re2/re2/re2.h"

#include <map>

#define MAX_CAPTURE_COUNT 36

namespace Falcon {


   /*
    @class RE
    @brief Engine bound regular expressions.
    @param pattern The regular expression pattern.
    @optparam options A string containing pattern options.
    @raise ParamError if the pattern string is malformed.

    The class can be initialized also through the gramar
    construct called R-String, which is a "r" letter followed
    by a single or double quote.

    Some operators are overloaded with special meanings:

    - The division operator "/" matches the string;
    - the multiply operator "*" matches the string, but requires a complete match;
    - the modulo operator "%" generates a list of captured expressions;
    - the power operator "**" returns the matched substring.

    @code
    > r"H...o" / "Hello World"   // true
    > r"H...o" * "Hello World"   // false
    > r"w.." ** "Hello World"    // "Wor"
    > (r'(\w*) (\w*)' % "Hello world").describe() //["Hello", "world"]
    @endcode

    @prop captures Number of capture expressions (excluding the total one).
    @prop groupNames Dictionary of named captured expressions and positions.
    @prop caseSensitive True to set case sensitivity match (the default),
          false to make it insensitive.
    @prop pattern A copy of the original pattern, as a (mutable) string.
    */

//=====================================================================
// Properties
//


ItemArray* internal_make_grab_array( re2::StringPiece* captured, int cc, bool grabAll )
{
   ItemArray* capt = new ItemArray;
   capt->reserve( cc );

   // skip the global match
   for( int i = grabAll ? 0 : 1; i < cc; ++i )
   {
      String* scapt = new String;
      scapt->fromUTF8( captured[i].data(), captured[i].size() );
      capt->append( FALCON_GC_HANDLE( scapt ) );
   }
   return capt;

}

static ItemArray* internal_grab( String* target, void * instance, bool grabAll )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );

   re2::StringPiece captured[MAX_CAPTURE_COUNT];
   int cc = re->NumberOfCapturingGroups() + 1;
   // paranoid...
   if( cc > MAX_CAPTURE_COUNT )
   {
      cc = MAX_CAPTURE_COUNT;
   }

   re2::StringPiece text(*target);

   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              captured, cc);

   if( match )
   {
      return internal_make_grab_array( captured, cc, grabAll );
   }

   return 0;
}


static void get_captures( const Class*, const String&, void* instance, Item& value )
{
   value.setInteger( static_cast<re2::RE2*>( instance )->NumberOfCapturingGroups() );
}


static void get_groupNames( const Class*, const String&, void* instance, Item& value )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );
   const std::map<std::string, int>& names = re->NamedCapturingGroups();
   std::map<std::string, int>::const_iterator iter = names.begin();
   std::map<std::string, int>::const_iterator end = names.end();

   ItemDict* dict = new ItemDict;

   while( iter != end )
   {
      String* name = new String;
      name->fromUTF8( iter->first.c_str() );
      int64 pos = iter->second;
      dict->insert( FALCON_GC_HANDLE(name), pos );
      ++iter;
   }

   value = FALCON_GC_HANDLE(dict);
}


static void get_caseSensitive( const Class*, const String&, void* instance, Item& value )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );
   value.setBoolean(re->options().case_sensitive());
}


static void get_pattern( const Class*, const String&, void* instance, Item& value )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );
   String* string = new String;
   try {
      string->fromUTF8(re->pattern().c_str());
      value = FALCON_GC_HANDLE( string );
   }
   catch( ... )
   {
      delete string;
      value.setNil();
   }
}

namespace {
/*#
 @method match RE
 @param target The string to be matched.
 @return True if the string matches somewhere with the
 regular expression.
*/
FALCON_DECLARE_FUNCTION( match, "target:S" );

/*#
 @method grab RE
 @param target The string to be matched.
 @return If the string matches, returns the matched content,
    otherwise returns nil
*/
FALCON_DECLARE_FUNCTION( grab, "target:S" );


/*#
 @method find RE
 @brief Returns the the first position where the pattern is matching.
 @param target The string to be matched.
 @optparam begin_ Where to begin the search in the target string.
 @optparam end_ Where to end the target string.
 @return If the regex is found in the @b target, returns the
    position where the pattern is matched, otherwise returns -1.

 If begin and end are set, the search is performed in the open
 range [begin_, end_[. If end is not set, it goes up to the end of
 the string.
*/
FALCON_DECLARE_FUNCTION( find, "target:S,begin_:[N],end_[N]" );

/*#
 @method findAll RE
 @param target The string to be matched.
 @brief Returns the all the initial positions where the pattern is matching.
 @optparam begin_ Where to begin the search in the target string.
 @optparam end_ Where to end the target string.
 @return If the regex is found in the @b target, returns the
    an array containing all the initial positions of the matches,
    otherwise returns nil

 If begin and end are set, the search is performed in the open
 range [begin, end[. If end is not set, it goes up to the end of
 the string.
*/
FALCON_DECLARE_FUNCTION( findAll, "target:S,begin_:[N],end_:[N]" );


/*#
 @method range RE
 @brief Returns the first range where the pattern is matching.
 @param target The string to be matched.
 @optparam begin_ Where to begin the search in the target string.
 @optparam end_ Where to end the target string.
 @return If the regex is found in the @b target, returns a range
    with the being-end position of the matched partern, otherwise
    returns nil

 If begin and end are set, the search is performed in the open
 range [begin, end[. If end is not set, it goes up to the end of
 the string.
*/
FALCON_DECLARE_FUNCTION( range, "target:S,begin_:[N],end_[N]" );

/*#
 @method rangeAll RE
 @brief Returns all the ranges where the pattern is matching.
 @param target The string to be matched.
 @optparam begin_ Where to begin the search in the target string.
 @optparam end_ Where to end the target string.
 @return If the regex is found in the @b target, returns the
    an array containing a set of ranges with all the found
    matches.

 If begin and end are set, the search is performed in the open
 range [begin, end[. If end is not set, it goes up to the end of
 the string.

 @note The found ranges cannot be overlapping.
*/
FALCON_DECLARE_FUNCTION( rangeAll, "target:S,begin_:[N],end_[N]" );

/*#
 @method capture RE
 @brief Returns the captured expressions.
 @param target The string to be matched.
 @optparam getAll If true, the first returned element is the whole match,
    otherwise, only parenthezized expressions are returned.
 @return If the string matches, returns an array with the captured expressions,
*/
FALCON_DECLARE_FUNCTION( capture, "target:S,getAll:[B]" );

/*#
 @method replace RE
 @brief Substitutes all the occurrences of the pattern.
 @param target The string where the replacement is done.
 @param replacer The replaced string.
 @return On success, a new copy of the string with the required
    substitution performed; on failure, returns nil.

 This method finds all the occurrences of the regular expression
 in @b target, and replaces them with the @b replacer.

 The replace may contain backslash expressions in form of
 '\\n' where @b n is a number indicating the nth capture (parenthesis)
 expression in the regular expression. \\0 represents the whole match,
 \\1 the first captured expression, \\2 the second and so on,
 up to @a RE.captures.
*/

FALCON_DECLARE_FUNCTION( replace, "target:S,replacer:S" );

/*#
 @method replaceFirst RE
 @brief Changes the first occurrence of the pattern.
 @param target The string where the replacement is done.
 @param replacer The replaced string.
 @return On success, a new copy of the string with the required
    substitution performed; on failure, returns nil.

 This method finds just the first occurrence of the regular expression
 in @b target, and replaces it with the @b replacer.

 The replace may contain backslash expressions in form of
 '\\n' where @b n is a number indicating the nth capture (parenthesis)
 expression in the regular expression. \\0 represents the whole match,
 \\1 the first captured expression, \\2 the second and so on,
 up to @a RE.captures.
*/

FALCON_DECLARE_FUNCTION( replaceFirst, "target:S,replacer:S" );

/*#
 @method substitute RE
 @brief Returns a transformation of the found pattern.
 @param target The string where the replacement is done.
 @param replacer The replaced string.
 @return On success, a new copy of the string with the required
    substitution performed; on failure, returns nil.

 This method finds a first occurrence of this regular expression
 in the @target string, and if found, returns a copy of the \b replacer.

 The replacer may contain backslash expressions in form of
 '\\n' where @b n is a number indicating the nth capture (parenthesis)
 expression in the regular expression. \\0 represents the whole match,
 \\1 the first captured expression, \\2 the second and so on,
 up to @a RE.captures.
*/

FALCON_DECLARE_FUNCTION( substitute, "target:S,replacer:S" );

/*#
 @method change RE
 @brief changes all the occurrences of the pattern in place.
 @param target The string where the replacement is done.
 @param replacer The replaced string.
 @return On success, true, on error, false.

 This method finds all the occurrences of the regular expression
 in @b target, and replaces them with the @b replacer.

 In case of success, The @b target string is modified in place.

 The replace may contain backslash expressions in form of
 '\\n' where @b n is a number indicating the nth capture (parenthesis)
 expression in the regular expression. \\0 represents the whole match,
 \\1 the first captured expression, \\2 the second and so on,
 up to @a RE.captures.
*/
FALCON_DECLARE_FUNCTION( change, "target:S,replacer:S" );

/*#
 @method changeFirst RE
 @brief Changes the first occurence of the pattern in place.
 @param target The string where the replacement is done.
 @param replacer The replaced string.
 @return On success, true, on error, false.

 This method finds just the first occurrence of the regular expression
 in @b target, and replaces it with the @b replacer.

 In case of success, The @b target string is modified in place.

 The replace may contain backslash expressions in form of
 '\\n' where @b n is a number indicating the nth capture (parenthesis)
 expression in the regular expression. \\0 represents the whole match,
 \\1 the first captured expression, \\2 the second and so on,
 up to @a RE.captures.
*/
FALCON_DECLARE_FUNCTION( changeFirst, "target:S,replacer:S" );

/*#
 @method chop RE
 @brief extract the matched pattern in place.
 @param target The string where the replacement is done.
 @param replacer The replaced string.
 @return On success, true, on error, false.

       This method finds a first occurrence of this regular expression
 in the @target string, and if found, it chops down the @b target string
 as indicated in the @b replacer.

 @note On success, the @b target string is changed in place.

 The replacer may contain backslash expressions in form of
 '\\n' where @b n is a number indicating the nth capture (parenthesis)
 expression in the regular expression. \\0 represents the whole match,
 \\1 the first captured expression, \\2 the second and so on,
 up to @a RE.captures.
*/

FALCON_DECLARE_FUNCTION( chop, "target:S,replacer:S" );

/*#
 @method consume RE
 @brief Cut the string up to where the pattern is matched and
     return the captured expressions.
 @param target The string that is to be consumed.
 @optparam getAll if true, return also the full matched pattern,
    otherwise just return the captured expressions.
 @return on success, an array with the matched captured patterns, or
 true if there aren't pattern to be captured. On failure, returns false.
*/
FALCON_DECLARE_FUNCTION( consume, "target:S,getAll:[B]" );

/*#
 @method consume RE
 @brief Cut the string up to where the pattern is matched,
    and return the matched pattern.
 @param target The string that is to be consumed.
 @return on success, the matched pattern. On failure, nil.
*/
FALCON_DECLARE_FUNCTION( consumeMatch, "target:S" );


void Function_match::invoke( VMContext* ctx, int32 )
{
   Item* i_target = ctx->param(0);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   re2::StringPiece text(*target);

   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              0, 0);

   ctx->returnFrame( Item().setBoolean(match) );
}


void Function_grab::invoke( VMContext* ctx, int32 )
{
   Item* i_target = ctx->param(0);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   re2::StringPiece text(*target);

   re2::StringPiece captured;
   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              &captured, 1);

   if( match )
   {
      String* ret = new String;
      ret->fromUTF8(captured.data(), captured.size() );
      ctx->returnFrame(FALCON_GC_HANDLE(ret));
   }
   else {
      ctx->returnFrame();
   }
}

bool internal_check_params( VMContext* ctx, String* &target, int64& start, int64& end )
{

   Item* i_target = ctx->param(0);
   Item* i_start = ctx->param(1);
   Item* i_end = ctx->param(2);

   if( i_target == 0|| ! i_target->isString() )
   {
      return false;
   }

   target = i_target->asString();
   start = 0;
   int64 tlen = target->length();
   end = tlen;

   if( i_start != 0 )
   {
      if( i_start->isOrdinal() )
      {
         start = i_start->forceInteger();
         /* Do not sanitize, as substring will do for us
         if ( start < 0 )
         {
            start = tlen + end;
         }
         if ( start > (int64) tlen )
         {
            start = tlen;
         }
         */
      }
      else if ( ! i_start->isNil() )
      {
         return false;
      }
   }

   if( i_end != 0 )
   {
      if( i_end->isOrdinal() )
      {
         end = i_end->forceInteger();
         /* Do not sanitize, as substring will do for us
         if ( end < 0 )
         {
            end = tlen + end;
         }
         if ( end > (int64) tlen )
         {
            end = tlen;
         }
         */

      }
      else if ( ! i_end->isNil() )
      {
         return false;
      }
   }

   return true;
}


static bool internal_find( VMContext* ctx, Item& result, bool (*on_found)(Item& target, int count, int pos, int size) )
{
   String* target;
   int64 start, end;
   if( ! internal_check_params( ctx, target, start, end ) )
   {
      return false;
   }

   String temp;
   if( start != 0 || end != target->length() )
   {
      // TODO: Use a UTF8 Converter instead.
      // In that case, remember to sanitize the values.
      temp = target->subString((int32)start, (int32)end);
      target = &temp;
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   re2::StringPiece text(*target);

   re2::StringPiece captured;
   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              &captured, 1);

   int count = 0;
   while( match )
   {
      uint32 pos_utf8 = captured.begin() - text.begin();
      uint32 size_utf8 = captured.end() - captured.begin();

      int64 pos = String::UTF8Size( text.data(), pos_utf8 );
      int64 size = String::UTF8Size( captured.data(), size_utf8 );

      if( ! on_found( result, count++, (int) (start + pos), (int) size ) )
      {
         break;
      }

      int next = static_cast<int>( pos + size + 1 );
      if( next >= text.size() )
      {
         break;
      }

      match = re->Match(text, next, text.size(),
                    re2::RE2::UNANCHORED,
                    &captured, 1);
   }

   return true;
}


bool on_found_find(Item& target, int , int pos, int )
{
   target = (int64) pos;
   return false;
}


bool on_found_findAll(Item& target, int count, int pos, int )
{
   if( count == 0 )
   {
      target = FALCON_GC_HANDLE(new ItemArray);
   }

   ItemArray* array = static_cast<ItemArray*>(target.asInst());
   array->append( pos );

   return true;
}

bool on_found_range(Item& target, int, int pos, int size)
{
   static class Class* crng = Engine::handlers()->rangeClass();
   Range* rng = static_cast<Range*>( crng->createInstance() );

   rng->start( pos );
   rng->end( pos + size );

   target = FALCON_GC_STORE( crng, rng );
   return false;
}


bool on_found_rangeAll(Item& target, int count, int pos, int size)
{
   static class Class* crng = Engine::handlers()->rangeClass();

   if( count == 0 )
   {
      target = FALCON_GC_HANDLE(new ItemArray);
   }
   ItemArray* array = static_cast<ItemArray*>(target.asInst());

   Range* rng = static_cast<Range*>( crng->createInstance() );
   rng->start( pos );
   rng->end( pos + size );
   array->append( FALCON_GC_STORE( crng, rng ) );

   return true;
}


void Function_find::invoke( VMContext* ctx, int32 )
{
   Item result = -1;

   if( ! internal_find(ctx, result, &on_found_find) ) {
      throw paramError();
   }

   ctx->returnFrame(result);
}


void Function_findAll::invoke( VMContext* ctx, int32 )
{
   Item result;

   if( ! internal_find(ctx, result, &on_found_findAll) ) {
      throw paramError();
   }

   ctx->returnFrame(result);
}


void Function_range::invoke( VMContext* ctx, int32 )
{
   Item result;

   if( ! internal_find(ctx, result, &on_found_range) ) {
      throw paramError();
   }

   ctx->returnFrame(result);
}

void Function_rangeAll::invoke( VMContext* ctx, int32 )
{
   Item result;

   if( ! internal_find(ctx, result, &on_found_rangeAll) ) {
      throw paramError();
   }

   ctx->returnFrame(result);
}


void Function_capture::invoke( VMContext* ctx, int32 )
{
   Item* i_target = ctx->param(0);
   Item* i_getall = ctx->param(1);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   bool bGetAll = i_getall != 0 ? i_getall->isTrue() : false;
   ItemArray* capt = internal_grab(i_target->asString(), ctx->self().asInst(), bGetAll );
   if( capt != 0 )
   {
      ctx->returnFrame( FALCON_GC_HANDLE(capt) );
   }
   else
   {
      ctx->returnFrame();
   }
}


static bool internal_change( VMContext* ctx, int mode )
{
   Item* i_target = ctx->param(0);
   Item* i_replacer = ctx->param(1);

   if( i_target == 0|| ! i_target->isString() || i_replacer == 0 || ! i_replacer->isString() )
   {
      return false;
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   if( mode >= 3 &&  target->isImmutable() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_acc_forbidden, .extra("Immutable string"));
   }
   String* replacer = i_replacer->asString();

   String* ret = 0;
   bool result = false;
   switch( mode )
   {
   case 0: // substitute
      ret = new String;
      result = re2::RE2::Extract(*target, *re, *replacer, *ret );
      break;

   case 1: // replace
      ret = new String(*target);
      result = re2::RE2::GlobalReplace(*ret, *re, *replacer ) > 0;
      break;

   case 2: // replace first
      ret = new String(*target);
      result = re2::RE2::Replace(*ret, *re, *replacer );
      break;

   case 3: // chop
      result = re2::RE2::Extract(*target, *re, *replacer, *target );
      //re2::RE2::Replace(*target, *re, *replacer );
      break;

   case 4: // change
      result = re2::RE2::GlobalReplace(*target, *re, *replacer ) > 0;
      break;

   case 5: // changeFirst
      result = re2::RE2::Replace(*target, *re, *replacer );
      break;
   }

   if( result )
   {
      if( ret != 0 )
      {
         ctx->returnFrame(FALCON_GC_HANDLE(ret));
      }
      else {
         ctx->returnFrame(Item().setBoolean(true));
      }
   }
   else
   {
      delete ret;
      if( mode >= 3 )
      {
         ctx->returnFrame(Item().setBoolean(false));
      }
      else {
         ctx->returnFrame();
      }
   }

   return true;
}


void Function_substitute::invoke( VMContext* ctx, int32 )
{
   if( ! internal_change( ctx, 0 ) )
   {
      throw paramError();
   }
}


void Function_replace::invoke( VMContext* ctx, int32 )
{
   if( ! internal_change( ctx, 1 ) )
   {
      throw paramError();
   }
}


void Function_replaceFirst::invoke( VMContext* ctx, int32 )
{
   if( ! internal_change( ctx, 2 ) )
   {
      throw paramError();
   }
}


void Function_chop::invoke( VMContext* ctx, int32 )
{
   if( !  internal_change( ctx, 3 ) )
   {
      throw paramError();
   }
}


void Function_change::invoke( VMContext* ctx, int32 )
{
   if( ! internal_change( ctx, 4 ) )
   {
      throw paramError();
   }
}


void Function_changeFirst::invoke( VMContext* ctx, int32 )
{
   if( ! internal_change( ctx, 4 ) )
   {
      throw paramError();
   }
}

void Function_consume::invoke( VMContext* ctx, int32 )
{
   Item* i_target = ctx->param(0);
   Item* i_getAll = ctx->param(1);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   bool getAll = i_getAll != 0 && i_getAll->isTrue();

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();
   if( target->isImmutable() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_acc_forbidden, .extra("Immutable string"));
   }

   re2::StringPiece captured[MAX_CAPTURE_COUNT];
   int cc = re->NumberOfCapturingGroups() + 1;
   // paranoid...
   if( cc > MAX_CAPTURE_COUNT )
   {
      cc = MAX_CAPTURE_COUNT;
   }

   String* cfr = i_target->asString();
   re2::StringPiece text(*cfr);

   bool match = re->Match(text,
              0,
              text.size(),
              re2::RE2::UNANCHORED,
              captured,
              cc);

   if( match )
   {
      int consumed = captured[0].end() - text.begin();

      if( cc  > 1 || getAll )
      {
         ItemArray* res = internal_make_grab_array( captured, cc, getAll );

         // change the string after, as text and captured are based on that
         length_t removed = String::UTF8Size( text.data(), consumed );
         target->remove(0, removed);

         ctx->returnFrame(FALCON_GC_HANDLE(res));
      }
      else {
         ctx->returnFrame(Item().setBoolean(true));
      }
   }
   else
   {
      ctx->returnFrame(Item().setBoolean(false));
   }
}


void Function_consumeMatch::invoke( VMContext* ctx, int32 )
{
   Item* i_target = ctx->param(0);

   if( i_target == 0|| ! i_target->isString() )
   {
      throw paramError();
   }

   re2::RE2* re = static_cast<re2::RE2*>(ctx->self().asInst());
   String* target = i_target->asString();

   if( target->isImmutable() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_acc_forbidden, .extra("Immutable string"));
   }

   re2::StringPiece captured;

   String* cfr = i_target->asString();
   re2::StringPiece text(*cfr);

   bool match = re->Match(text, 0, text.size(),
              re2::RE2::UNANCHORED,
              &captured, 1);

   if( match )
   {
      int consumed = captured.end() - text.begin();

      String* res = new String;
      res->fromUTF8(captured.data(), captured.length());

      // change the string after, as text and captured are based on that
      length_t removed = String::UTF8Size( text.data(), consumed );
      target->remove(0, removed);

      ctx->returnFrame(FALCON_GC_HANDLE(res));
   }
   else
   {
      ctx->returnFrame();
   }
}
}

//======================================================================
//
// Class properties used for enumeration
//

ClassRE::ClassRE():
   Class( "RE", FLC_CLASS_ID_RE )
{
   addProperty( "captures", &get_captures );
   addProperty( "caseSensitive", &get_caseSensitive );
   addProperty( "groupNames", &get_groupNames );
   addProperty( "pattern", &get_pattern );

   addMethod(new Function_match);
   addMethod(new Function_grab);
   addMethod(new Function_find);
   addMethod(new Function_findAll);
   addMethod(new Function_range);
   addMethod(new Function_rangeAll);
   addMethod(new Function_capture);

   addMethod(new Function_replace);
   addMethod(new Function_replaceFirst);
   addMethod(new Function_substitute);
   addMethod(new Function_change);
   addMethod(new Function_changeFirst);
   addMethod(new Function_chop);

   addMethod(new Function_consume);
   addMethod(new Function_consumeMatch);
}


ClassRE::~ClassRE()
{
}


int64 ClassRE::occupiedMemory( void* instance ) const
{
   re2::RE2* re2 = static_cast<re2::RE2*>( instance );
   // TODO: REASONABILY precise measurement.
   return sizeof(re2::RE2) + re2->ProgramSize() + 32;
}


void ClassRE::dispose( void* self ) const
{
   delete static_cast<re2::RE2*>( self );
}


void* ClassRE::clone( void* instance ) const
{
   re2::RE2* re2 = static_cast<re2::RE2*>( instance );
   re2::RE2* copy = new re2::RE2( re2->pattern(), re2->options() );

   return copy;
}

void* ClassRE::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassRE::store( VMContext*, DataWriter* stream, void* instance ) const
{
   re2::RE2* re2 = static_cast<re2::RE2*>( instance );
   TRACE2( "ClassRE::store -- \"%s\"", re2->pattern().c_str() );
   String pattern;
   pattern.fromUTF8(re2->pattern().c_str());
   stream->write( pattern );
   stream->write( re2->options().case_sensitive() );
   stream->write( re2->options().longest_match() );
   stream->write( re2->options().one_line() );
   stream->write( re2->options().never_nl() );

}


void ClassRE::restore( VMContext* ctx, DataReader* dr ) const
{
   static Class* classRe = Engine::handlers()->reClass();

   String pattern;
   bool cs;
   bool longest;
   bool one_line;
   bool never_nL;

   dr->read( pattern );
   dr->read( cs );
   dr->read( longest );
   dr->read( one_line );
   dr->read( never_nL );

   TRACE2( "ClassRE::restore -- \"%s\"", pattern.c_ize() );
   re2::RE2::Options opts;
   opts.set_case_sensitive(cs);
   opts.set_longest_match(longest);
   opts.set_one_line(one_line);
   opts.set_never_nl(never_nL);

   re2::RE2* re2 = new re2::RE2(pattern, opts);
   // let's trust our source.
   ctx->pushData( Item( classRe, re2 ) );
}


void ClassRE::describe( void* instance, String& target, int, int ) const
{
   re2::RE2* self = static_cast<re2::RE2*>( instance );

   String temp;
   temp.fromUTF8( self->pattern().c_str() );
   temp.escapeQuotes();
   if( self->options().case_sensitive() )
   {
      target = "r";
   }
   else {
      target = "R";
   }

   target += "'" + temp + "'";

   if( self->options().one_line() )
   {
      target += "o";
   }

   if( self->options().longest_match() )
   {
      target += "l";
   }

   if( self->options().never_nl() )
   {
      target += "n";
   }

}

void ClassRE::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<re2::RE2*>( instance )->gcMark( mark );
}


bool ClassRE::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<re2::RE2*>( instance )->currentMark() >= mark;
}

//=======================================================================
// Operands
//

bool ClassRE::op_init( VMContext* ctx, void*, int pcount ) const
{
   // no param?
   String* sopts = 0;
   if( pcount == 0 )
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                              .extra("S,[S]"));
   }

   Item* params = ctx->opcodeParams(pcount);

   if( pcount >= 2 )
   {
      Item& i_opts = params[1];
      if( !(i_opts.isString() || i_opts.isNil()) )
      {
         throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                        .extra("S,[S]"));
      }
      else if( i_opts.isString() )
      {
         sopts = i_opts.asString();
      }
   }

   Item& i_pattern = params[0];
   if ( ! i_pattern.isString())
   {
      throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                              .extra("S,[S]"));
   }
   
   String* pattern = i_pattern.asString();
   re2::RE2::Options opts;

   if( sopts != 0 )
   {
      for( uint32 i = 0; i< sopts->length(); ++i )
      {
         uint32 chr = sopts->getCharAt(i);
         switch( chr )
         {
         case 'i': opts.set_case_sensitive(false); break;
         case 'n': opts.set_never_nl(true); break;
         case 'l': opts.set_longest_match(true); break;
         case 'o': opts.set_one_line(true); break;
         default:
            throw new ParamError(ErrorParam(e_inv_params, __LINE__, SRC)
                           .extra("Unrecognized options: "+*sopts));
         }
      }
   }

   re2::RE2* re = new re2::RE2( *pattern, opts );

   if( ! re->ok() )
   {
      String error;
      String temp;

      error.fromUTF8( re->error().c_str() );
      temp.fromUTF8( re->error_arg().c_str() );
      error += " at ";
      error += temp;
      delete re;

      throw new ParamError(ErrorParam(e_regex_def, __LINE__, SRC)
                    .extra(error));
   }

   ctx->opcodeParam(pcount).setUser( this, re );

   return false;
}


static void internal_match( VMContext* ctx, void* instance, bool partial )
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );

   if( ! ctx->topData().isString() )
   {
     throw new OperandError(ErrorParam(e_invalid_op, __LINE__, SRC )
              .extra("S"));
   }

   String* cfr = ctx->topData().asString();
   re2::StringPiece text(*cfr);
   bool match = re->Match(text, 0, text.size(),
                partial ? re2::RE2::UNANCHORED : re2::RE2::ANCHOR_BOTH,
                0, 0);

   ctx->popData();
   ctx->topData().setBoolean( match );
}

void ClassRE::op_compare( VMContext* ctx, void* ) const
{
   Item *op1, *op2;

   ctx->operands( op1, op2 );

   // RE has type digntity, it's compare is managed by the standard item compare.
   ctx->stackResult( 2, (int64)  op1->compare(*op2) );
}

void ClassRE::op_div( VMContext* ctx, void* instance ) const
{
   internal_match( ctx, instance, true );
}

void ClassRE::op_mod( VMContext* ctx, void* instance ) const
{

   if( ! ctx->topData().isString() )
   {
     throw new OperandError(ErrorParam(e_invalid_op, __LINE__, SRC )
              .extra("S"));
   }

   Item ret;
   String* cfr = ctx->topData().asString();
   ItemArray* capt = internal_grab( cfr, instance, false);
   if( capt != 0 )
   {
      ret = FALCON_GC_HANDLE( capt );
   }

   ctx->popData();
   ctx->topData() = ret;
}

void ClassRE::op_mul( VMContext* ctx, void* instance ) const
{
   internal_match( ctx, instance, false );
}

void ClassRE::op_pow( VMContext* ctx, void* instance ) const
{
   re2::RE2* re = static_cast<re2::RE2*>( instance );

   if( ! ctx->topData().isString() )
   {
     throw new OperandError(ErrorParam(e_invalid_op, __LINE__, SRC )
              .extra("S"));
   }

   String* cfr = ctx->topData().asString();
   re2::StringPiece piece;

   re2::StringPiece text(*cfr);
   bool match = re->Match(text,
              0,
              text.size(),
              re2::RE2::UNANCHORED,
              &piece,
              1);

   Item ret;
   if( match )
   {
      String *captured = new String;
      captured->fromUTF8( piece.data(), piece.size() );
      ret = FALCON_GC_HANDLE( captured );
   }

   ctx->popData();
   ctx->topData() = ret;
}
}

/* end of classre.cpp */
