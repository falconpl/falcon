/*
   FALCON - The Falcon Programming Language.
   FILE: string.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven nov 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

/** \file
   Short description
*/


#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/membuf.h>

#include <string.h>

/*#
   @funset core_string_functions String functions
   @brief Functions manipulating strings

   @beginset core_string_functions
*/
namespace Falcon {
namespace core {

static void process_strFrontBackParams( VMachine *vm, String* &str, bool &bNumeric, bool &bRemove, int32 &len )
{
   Item *i_count;

   if ( vm->self().isMethodic() )
   {
      str = vm->self().asString();
      i_count = vm->param(0);
      bRemove = vm->param(1) != 0 && vm->param(1)->isTrue();
      bNumeric = vm->param(2) != 0 && vm->param(2)->isTrue();
   }
   else
   {
      Item *i_str = vm->param(0);
      if( i_str == 0 || ! i_str->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,[N,B,B]" ) );
      }

      str = i_str->asString();
      i_count = vm->param(1);
      bRemove = vm->param(2) != 0 && vm->param(2)->isTrue();
      bNumeric = vm->param(3) != 0 && vm->param(3)->isTrue();
   }

   if ( i_count == 0 || i_count->isNil() )
   {
      len = 1;
   }
   else if ( ! i_count->isOrdinal()  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,[N,B,B]" ) );
   }
   else {
      len = (int32) i_count->forceInteger();
   }
}

/*#
   @method charSize String
   @brief Returns or changes the size in bytes in this string.
   @optparam bpc New value for bytes per character (1, 2 or 4).
   @return This string if @b bpc is given, or the current character size value if not given.
*/

FALCON_FUNC String_charSize( VMachine *vm )
{
   Item *i_bpc = vm->param(0);
   String* str = vm->self().asString();

   if ( i_bpc == 0 )
   {
      // no parameters -- just read us.
      vm->retval( (int64) str->manipulator()->charSize() );
      return;
   }
   else if( ! i_bpc->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "[N]" ) );
   }

   uint32 bpc = (uint32) i_bpc->forceInteger();
   if ( ! str->setCharSize( bpc ) )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ) );
   }

   vm->retval( str );
}

/*#
   @method front String
   @brief Returns the first character in a string.
   @optparam count Number of characters to be returned (1 by default).
   @optparam numeric If true, returns a character value instead of a string.
   @optparam remove If true, remove also the character.
   @return The first element or nil if the string is empty.

   This method returns a string containing one character from the beginning of the string,
   or eventually more characters in case a number > 1 is specified in @b count.

   If @b remove is true, then the character is removed and the string is shrunk.
   @see strFront

   If @b numeric is true,
   the UNICODE value of the string character will be returned, otherwise the caller
   will receive a string containing the desired character. In this case, @b count
   parameter will be ignored and only one UNICODE value will be returned.
*/

/*#
   @function strFront
   @brief Returns the first character in a string.
   @param str The string on which to operate.
   @optparam count Number of characters to be taken (defaults to 1).
   @optparam remove If true, remove also the character.
   @optparam numeric If true, returns a character value instead of a string.
   @return The first element or nil if the string is empty.

   This function returns a string containing one character from the beginning of @b str,
   or eventually more characters in case a number > 1 is specified in @b count.

   If @b remove is true, then the character is removed and the string is shrunk.
   @see String.front

   If @b numeric is true,
   the UNICODE value of the string character will be returned, otherwise the caller
   will receive a string containing the desired character. In this case, @b count
   parameter will be ignored and only one UNICODE value will be returned.
*/

FALCON_FUNC mth_strFront( VMachine *vm )
{
   String *str;
   bool bNumeric;
   bool bRemove;
   int32 len;

   process_strFrontBackParams( vm, str, bNumeric, bRemove, len );

   if ( bNumeric )
   {
      if ( str->size() == 0 )
      {
         vm->retnil();
      }
      else
      {
         vm->retval( (int64) str->getCharAt( 0 ) );

         if( bRemove )
            str->remove( 0, 1 );
      }
   }
   else
   {
      if ( len <= 0 ) {
         vm->retval( new CoreString( "" ) );
      }
      else if ( len > (int32) str->length() ) {
         vm->retval( new CoreString( *str ) );
      }
      else {
         vm->retval( new CoreString( *str, 0, len ) );
      }

      if( bRemove )
            str->remove(0, len );
   }
}

/*#
   @method back String
   @brief Returns the first character in a string.
   @optparam count Number of characters to be taken (defaults to 1).
   @optparam numeric If true, returns a character value instead of a string.
   @optparam remove If true, remove also the character.
   @return The first element or nil if the string is empty.

   This function returns a string containing one character from the end of this string,
   or eventually more characters in case a number > 1 is specified in @b count.

   If @b remove is true, then the character is removed and the string is shrunk.
   @see strFront

   If @b numeric is true,
   the UNICODE value of the string character will be returned, otherwise the caller
   will receive a string containing the desired character. In this case, @b count
   parameter will be ignored and only one UNICODE value will be returned.
*/

/*#
   @function strBack
   @brief Returns the last character(s) in a string.
   @param str The string on which to operate.
   @optparam count Number of characters to be taken (defaults to 1).
   @optparam remove If true, remove also the character.
   @optparam numeric If true, returns a character value instead of a string.
   @return The first element or nil if the string is empty.

   This function returns a string containing one character from the end of @b str,
   or eventually more characters in case a number > 1 is specified in @b count.

   If @b remove is true, then the characters are removed and the string is shrunk.
   @see String.front

   If @b numeric is true,
   the UNICODE value of the string character will be returned, otherwise the caller
   will receive a string containing the desired character. In this case, @b count
   parameter will be ignored and only one UNICODE value will be returned.
*/

FALCON_FUNC mth_strBack( VMachine *vm )
{
   String *str;
   bool bNumeric;
   bool bRemove;
   int32 len;

   process_strFrontBackParams( vm, str, bNumeric, bRemove, len );

   int32 strLen = str->length();

   if ( bNumeric )
   {
      if ( str->size() == 0 )
      {
         vm->retnil();
      }
      else
      {
         vm->retval( (int64) str->getCharAt( strLen-1 ) );

         if( bRemove )
            str->remove( strLen-1, 1 );
      }
   }
   else
   {
      if ( len <= 0 ) {
         vm->retval( new CoreString( "" ) );
      }
      else if ( len >= strLen ) {
         vm->retval( new CoreString( *str ) );
      }
      else {
         vm->retval( new CoreString( *str, strLen-len ) );
      }

      if( bRemove )
         str->remove( strLen-len, len );

   }
}

/**
   @method first String
   @brief Returns an iterator to the head of this string.
   @return An iterator.
*/

/*FALCON_FUNC String_first( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   iterator->setProperty( "_pos", Item( 0 ) );
   iterator->setProperty( "_origin", vm->self() );
   vm->retval( iterator );
}*/

/**
   @method last String
   @brief Returns an iterator to the tail of this string.
   @return An iterator.
*/

/*FALCON_FUNC String_last( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   String *orig = vm->self().asString();
   iterator->setProperty( "_pos", Item( orig->size() == 0 ? 0 : (int64) orig->length() - 1 ) );
   iterator->setProperty( "_origin", vm->self() );
   vm->retval( iterator );
}*/


/*#
   @function strSplitTrimmed
   @brief Subdivides a string in an array of substrings given a token substring.
   @param string The string that must be split.
   @param token The token by which the string should be split.
   @optparam count Optional maximum split count.
   @return An array of strings containing the split string.

   This function returns an array of strings extracted from the given parameter.
   The array is filled with strings extracted from the first parameter, by dividing
   it based on the occurrences of the token substring. A count parameter may be
   provided to limit the splitting, so to take into consideration only the first
   relevant  tokens.  Adjacent matching tokens will be ignored.
   If no matches are possible, the returned array contains
   at worst a single element containing a copy of the whole string passed as a
   parameter.

   Contrarily to @a strSplit, this function will "eat up" adjacent tokens. While
   @a strSplit is more adequate to parse field-oriented strings (as i.e.
   colon separated fields in configuration files) this function is best employed
   in word extraction.

   @note this function is equivalent to the FBOM method String.splittr

   @note See @a Tokenizer for a more adequate function to scan extensively
   wide strings.
*/

/*#
   @method splittr String
   @brief Subdivides a string in an array of substrings given a token substring.
   @param token The token by which the string should be split.
   @optparam count Optional maximum split count.
   @return An array of strings containing the split string.

   @see strSplitTrimmed
*/

FALCON_FUNC  mth_strSplitTrimmed ( ::Falcon::VMachine *vm )
{
   Item *target;
   Item *splitstr;
   Item *count;

   // Parameter checking;
   if( vm->self().isMethodic() )
   {
      target = &vm->self();
      splitstr = vm->param(0);
      count = vm->param(1);
   }
   else
   {
      target = vm->param(0);
      splitstr = vm->param(1);
      count = vm->param(2);
   }

   uint32 limit;

   if ( target == 0 || ! target->isString()
        || (splitstr != 0 && ! (splitstr->isString() || splitstr->isNil()))
        || ( count != 0 && ! count->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "S, [N]" : "S, S, [N]" ) );
   }

   limit = count == 0 ? 0xffffffff: (int32) count->forceInteger();

   // Parameter extraction.
   String *tg_str = target->asString();
   uint32 tg_len = target->asString()->length();

   // split in chars?
   if( splitstr == 0 || splitstr->isNil() || splitstr->asString()->size() == 0)
   {
      // split the string in an array.
      if( limit > tg_len )
         limit = tg_len;

      CoreArray* ca = new CoreArray( limit );
      for( uint32 i = 0; i < limit - 1; ++i )
      {
         CoreString* elem = new CoreString(1);
         elem->append( tg_str->getCharAt(i) );
         ca->append( elem );
      }
      // add remains if there are any
      if(limit <= tg_len)
         ca->append(tg_str->subString(limit - 1));

      vm->retval( ca );
      return;
   }

   String *sp_str = splitstr->asString();
   uint32 sp_len = splitstr->asString()->length();

   // return item.
   CoreArray *retarr = new CoreArray;

   // if the split string is empty, return empty string
   if ( sp_len == 0 )
   {
      retarr->append( new CoreString() );
      vm->retval( retarr );
      return;
   }

   // if the token is wider than the string, just return the string
   if ( tg_len < sp_len )
   {
      retarr->append( new CoreString(  *tg_str ) );
      vm->retval( retarr );
      return;
   }

   uint32 pos = 0;
   uint32 last_pos = 0;
   bool lastIsEmpty = false;
   // scan the string
   while( limit > 1 && pos <= tg_len - sp_len  )
   {
      uint32 sp_pos = 0;
      // skip matching pattern-
      while ( tg_str->getCharAt( pos ) == sp_str->getCharAt( sp_pos ) &&
              sp_pos < sp_len && pos <= tg_len - sp_len ) {
         sp_pos++;
         pos++;
      }

      // a match?
      if ( sp_pos == sp_len ) {
         // put the item in the array.
         uint32 splitend = pos - sp_len;
         retarr->append( new CoreString( String( *tg_str, last_pos, splitend ) ) );

         lastIsEmpty = (last_pos >= splitend);

         last_pos = pos;
         limit--;
         // skip matching pattern
         while( sp_pos == sp_len && pos <= tg_len - sp_len ) {
            sp_pos = 0;
            last_pos = pos;
            while ( tg_str->getCharAt( pos ) == sp_str->getCharAt( sp_pos )
                    && sp_pos < sp_len && pos <= tg_len - sp_len ) {
               sp_pos++;
               pos++;
            }
         }
         pos = last_pos;

      }
      else
         pos++;
   }

   // Residual element?
   // -- but only if we didn't already put a "" in the array
   if ( limit >= 1 && ! lastIsEmpty ) {
      retarr->append( new CoreString( String( *tg_str, last_pos, (uint32) tg_len ) ) );
   }

   vm->retval( retarr );
}


/*#
   @function strSplit
   @brief Subdivides a string in an array of substrings given a token substring.
   @param string The string that must be split.
   @optparam token The token by which the string should be split.
   @optparam count Optional maximum split count.
   @return An array of strings containing the split string.

   This function returns an array of strings extracted from the given parameter.
   The array is filled with strings extracted from the first parameter, by dividing
   it based on the occurrences of the token substring. A count parameter may be
   provided to limit the splitting, so to take into consideration only the first
   relevant  tokens.  Adjacent matching tokens will cause the returned array to
   contains empty strings. If no matches are possible, the returned array contains
   at worst a single element containing a copy of the whole string passed as a
   parameter.

   For example, the following may be useful to parse a INI file where keys are
   separated from values by "=" signs:

   @code
   key, value = strSplit( string, "=", 2 )
   @endcode

   This code would return an array of 2 items at maximum; if no "=" signs are found
   in string, the above code would throw an error because of unpacking size, a
   thing that may be desirable in a parsing code. If there are more than one "=" in
   the string, only the first starting from left is considered, while the others
   are returned in the second item, unparsed.

   If the @b token is empty or not given, the string is returned as a sequence of
   1-character strings in an array.


   @note This function is equivalent to the fbom method @a String.split. The above
   example can be rewritten as:
   @code
   key, value = string.split( "=", 2 )
   @endcode
*/

/*#
   @method split String
   @brief Subdivides a string in an array of substrings given a token substring.
   @optparam token The token by which the string should be split.
   @optparam count Optional maximum split count.
   @return An array of strings containing the split string.


   @see strSplit
*/
FALCON_FUNC  mth_strSplit ( ::Falcon::VMachine *vm )
{
   Item *target;
   Item *splitstr;
   Item *count;

   // Parameter checking;
   if( vm->self().isMethodic() )
   {
      target = &vm->self();
      splitstr = vm->param(0);
      count = vm->param(1);
   }
   else
   {
      target = vm->param(0);
      splitstr = vm->param(1);
      count = vm->param(2);
   }

   if ( (target == 0 || ! target->isString())
        || (splitstr != 0 && ! (splitstr->isString() || splitstr->isNil()))
        || ( count != 0 && ! count->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "S, [N]" : "S, S, [N]" ) );
   }

   // Parameter extraction.
   String *tg_str = target->asString();
   uint32 tg_len = target->asString()->length();
   uint32 limit = count == 0 ? 0xffffffff: (int32) count->forceInteger();

   // split in chars?
   if( splitstr == 0 || splitstr->isNil() || splitstr->asString()->size() == 0)
   {
      // split the string in an array.
      if( limit > tg_len )
         limit = tg_len;

      CoreArray* ca = new CoreArray( limit );
      for( uint32 i = 0; i < limit - 1; ++i )
      {
         CoreString* elem = new CoreString(1);
         elem->append( tg_str->getCharAt(i) );
         ca->append( elem );
      }
      // add remains if there are any
      if(limit <= tg_len)
         ca->append(tg_str->subString(limit - 1));

      vm->retval( ca );
      return;
   }

   String *sp_str = splitstr->asString();
   uint32 sp_len = sp_str->length();


   // return item.
   CoreArray *retarr = new CoreArray;

   // if the split string is empty, return empty string
   if ( sp_len == 0 )
   {
      retarr->append( new CoreString() );
      vm->retval( retarr );
      return;
   }

   // if the token is wider than the string, just return the string
   if ( tg_len < sp_len )
   {
      retarr->append( new CoreString( *tg_str ) );
      vm->retval( retarr );
      return;
   }

   uint32 pos = 0;
   uint32 last_pos = 0;
   // scan the string
   while( limit > 1 && pos <= tg_len - sp_len  )
   {
      uint32 sp_pos = 0;
      // skip matching pattern-
      while ( tg_str->getCharAt( pos ) == sp_str->getCharAt( sp_pos ) &&
              sp_pos < sp_len && pos <= tg_len - sp_len ) {
         sp_pos++;
         pos++;
      }

      // a match?
      if ( sp_pos == sp_len ) {
         // put the item in the array.
         uint32 splitend = pos - sp_len;
         retarr->append( new CoreString( *tg_str, last_pos, splitend ) );
         last_pos = pos;
         limit--;

      }
      else
         pos++;
   }

   // Residual element?
   if ( limit >= 1 || last_pos < tg_len ) {
      uint32 splitend = tg_len;
      retarr->append( new CoreString( *tg_str, last_pos, splitend ) );
   }

   vm->retval( retarr );
}

/*#
   @function strMerge
   @brief  Merges an array of strings into a string.
   @param array An array of strings to be merged.
   @optparam mergeStr A string placed between each merge.
   @optparam count Maximum count of merges.
   @return The merged string.

   The function will return a new string containing the concatenation
   of the strings inside the array parameter. If the array is empty,
   an empty string is returned. If a mergeStr parameter is given, it
   is added to each pair of string merged; mergeStr is never added at
   the end of the new string. A count parameter may be specified to
   limit the number of elements merged in the array.

   The function may be used in this way:

   @code
   a = strMerge( [ "a", "string", "of", "words" ], " " )
   printl( a ) // prints "a string of words"
   @endcode

   If an element of the array is not a string, an error is raised.
*/

/*#
   @method merge String
   @brief Merges an array of strings into one.
   @param array An array of strings to be merged.
   @return This string.

   This method works as strMerge, using this string as
   separator between the strings in the array.

   The function may be used in this way:

   @code
   a = " ".merge( [ "a", "string", "of", "words" ] )
   printl( a ) // prints "a string of words"
   @endcode

   If an element of the array is not a string, an error is raised.
*/

FALCON_FUNC  mth_strMerge ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *source = vm->param(0);
   Item *mergestr = vm->param(1);
   Item *count = vm->param(2);
   uint64 limit;

   if ( source == 0 || ! source->isArray()
        || ( mergestr != 0 && ! mergestr->isString() )
        || ( count != 0 && ! count->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "A" : "A,[S],[N]" ) );
   }

   // Parameter estraction.

   limit = count == 0 ? 0xffffffff: count->forceInteger();

   String *mr_str;
   if( mergestr != 0 )
   {
      mr_str = mergestr->asString();
   }
   else
      mr_str = vm->self().isMethodic() ? vm->self().asString() : 0;

   Item *elements = source->asArray()->items().elements();
   uint32 len = source->asArray()->length();
   if ( limit < len )
      len = (uint32) limit;

   CoreString *ts = new CoreString;

   // filling the target.
   for( uint32 i = 1; i <= len ; i ++ )
   {
      Item* item = elements+(i-1);

      if ( item->isString() )
         *ts += *item->asString();
      else
      {
         String temp;
         vm->itemToString( temp, item );
         *ts += temp;
      }

      if ( mr_str != 0 && i < len )
         ts->append( *mr_str );

      ts->reserve( len/i * ts->size() );
   }

   vm->retval( ts );
}


/*#
   @method join String
   @brief Joins the parameters into a new string.
   @param ...
   @return A new string created joining the parameters, left to right.

   If this string is not empty, it is copied between each joined string.

   For example, the next code separates each value with ", "

   @code
   > ", ".join( "have", "a", "nice", "day" )
   @endcode

   If the parameters are not string, a standard @a toString conversion is tried.
*/

FALCON_FUNC  String_join ( ::Falcon::VMachine *vm )
{
   // Parameter extraction.
   CoreString *ts = new CoreString;
   String *self = vm->self().asString();
   uint32 pc = vm->paramCount();

   if ( pc > 0 )
   {
      Item *head = vm->param(0);
      if ( head->isString() )
         *ts = *head->asString();
      else
         vm->itemToString( *ts, head );

      ts->bufferize();

      for( uint32 i = 1; i < pc; i++ )
      {
         if( self->size() != 0 )
            *ts += *self;

         Item *item = vm->param(i);
         if ( item->isString() )
            *ts += *item->asString();
         else
         {
            String temp;
            vm->itemToString( temp, item );
            *ts += temp;
         }

         ts->reserve( pc/i * ts->size() );
      }
   }

   vm->regA().setString( ts );
}

/*#
   @function strFind
   @brief Finds a substring.
   @param string String where the search will take place.
   @param needle Substring to search for.
   @optparam start Optional position from which to start the search in string.
   @optparam end Optional position at which to end the search in string.
   @return The position where the substring is found, or -1.

   Returns the index in string were needle begins, or -1 if not present. Giving a
   start parameter will cause the search to start from the given position up to the
   end of the string; if a match can be made at start position, then the the
   returned value will be the same as start, so when repeating searches in a string
   for all the possible matches, repeat until the result is -1 by adding one to the
   returned value and using it as start position for the next search.

   If an end position is given, it is used as upper limit for the search, so that
   the search is in the interval [start, end-1].

   @note This function is equivalent to the fbom method String.find
*/

/*#
   @method find String
   @brief Finds a substring.
   @param needle Substring to search for.
   @optparam start Optional position from which to start the search in string.
   @optparam end Optional position at which to end the search in string.
   @return The position where the substring is found, or -1.

   @see strFind
*/
FALCON_FUNC  mth_strFind ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *needle = vm->param(1);
   Item *start_item = vm->param(2);
   Item *end_item = vm->param(3);

   if ( vm->self().isMethodic() )
   {
      target = &vm->self();
      needle = vm->param(0);
      start_item = vm->param(1);
      end_item = vm->param(2);
   }
   else
   {
      target = vm->param(0);
      needle = vm->param(1);
      start_item = vm->param(2);
      end_item = vm->param(3);
   }

   if ( target == 0 || ! target->isString()
        || needle == 0 || ! needle->isString()
        || (start_item != 0 && ! start_item->isOrdinal())
        || (end_item != 0 && ! end_item->isOrdinal())
        )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "S,[N],[N]" : "S,S,[N],[N]" ) );
   }

   int64 start = start_item == 0 ? 0 : start_item->forceInteger();
   int64 end = end_item == 0 ? csh::npos : end_item->forceInteger();
   String *sTarget = target->asString();

   // negative? -- fix
   if ( start < 0 ) end = sTarget->length() + start +1;

   // out of range?
   if ( start < 0 || start >= sTarget->length() )
   {
      vm->retval( -1 );
      return;
   }

   if ( end < 0 ) end = sTarget->length() + end+1;
   // again < than 0? -- it's out of range.
   if ( end < 0 )
   {
      vm->retval( -1 );
      return;
   }

   uint32 pos = target->asString()->find( *needle->asString(), (uint32) start, (uint32) end );
   if ( pos != csh::npos )
      vm->retval( (int64)pos );
   else
      vm->retval( -1 );
}

/*#
   @function strBackFind
   @brief Finds a substring backwards.
   @param string String where the search will take place.
   @param needle Substring to search for.
   @optparam start Optional position from which to start the search in string.
   @optparam end Optional position at which to end the search in string.
   @return The position where the substring is found, or -1.

   Works exactly as @a strFind, except for the fact that the last match
   in the string (or in the specified interval) is returned.
*/

/*#
   @method rfind String
   @brief Finds a substring backwards.
   @param needle Substring to search for.
   @optparam start Optional position from which to start the search in string.
   @optparam end Optional position at which to end the search in string.
   @return The position where the substring is found, or -1.

   @see strBackFind
*/

FALCON_FUNC  mth_strBackFind ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *needle = vm->param(1);
   Item *start_item = vm->param(2);
   Item *end_item = vm->param(3);

   if ( vm->self().isMethodic() )
   {
      target = &vm->self();
      needle = vm->param(0);
      start_item = vm->param(1);
      end_item = vm->param(2);
   }
   else
   {
      target = vm->param(0);
      needle = vm->param(1);
      start_item = vm->param(2);
      end_item = vm->param(3);
   }

   if ( target == 0 || ! target->isString()
        || needle == 0 || ! needle->isString()
        || (start_item != 0 && ! start_item->isOrdinal())
        || (end_item != 0 && ! end_item->isOrdinal())
        )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "S,[N],[N]" : "S,S,[N],[N]" ) );
   }

   int64 start = start_item == 0 ? 0 : (int64) start_item->forceInteger();
   int64 end = end_item == 0 ? csh::npos : (int64) end_item->forceInteger();
   String *sTarget = target->asString();

   // negative? -- fix
   if ( start < 0 ) end = sTarget->length() + start +1;

   // out of range?
   if ( start < 0 || start >= sTarget->length() )
   {
      vm->retval( -1 );
      return;
   }

   if ( end < 0 ) end = sTarget->length() + end+1;
   // again < than 0? -- it's out of range.
   if ( end < 0 )
   {
      vm->retval( -1 );
      return;
   }

   uint32 pos = target->asString()->rfind( *needle->asString(), (uint32) start, (uint32) end );
   if ( pos != csh::npos )
      vm->retval( (int)pos );
   else
      vm->retval( -1 );
}

/*#
   @method rtrim String
   @brief Trims trailing whitespaces in a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.
   @raise AccessError if the given item is not a string.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the end of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.
*/

/*#
   @method ftrim String
   @brief Trims front whitespaces in a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.
   @raise AccessError if the given item is not a string.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the beginning of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.

   @see strFrontTrim
*/

/*#
   @method trim String
   @brief Trims whitespaces from both ends of a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.
   @raise AccessError if the given item is not a string.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at both ends of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.

   @see strTrim
*/

/*#
   @function strTrim
   @brief Removes the white spaces at the beginning and at the end of a string.
   @param string The string to be trimmed.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed substring.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the end of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.
*/
FALCON_FUNC  mth_strTrim ( ::Falcon::VMachine *vm )
{
   String *self;
   Item *trimChars;

   if ( vm->self().isMethodic() )
   {
      self = vm->self().asString();
      trimChars = vm->param(0);
   }
   else
   {
      Item *i_str = vm->param( 0 );
      if ( i_str == 0 || ! i_str->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      self = i_str->asString();
      trimChars = vm->param(1);
   }

   CoreString *cs = new CoreString( *self );
   cs->garbage().mark( vm->generation() );

   if ( trimChars == 0 || trimChars->isNil() ) {
      cs->trim();
      vm->retval( cs );
   }
   else if ( ! trimChars->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   else
   {
      String *trim = trimChars->asString();
      int32 tLen = trim->length();
      int32 len = cs->length();
      int32 start = 0;
      int32 end = len;
      uint32 chr;
      int found = 0;

      while( start < len )
      {
         found = 0;
         chr = cs->getCharAt( start );
         for ( int32 tIdx=0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         start++;
      }

      while( end > start )
      {
         found = 0;
         chr = cs->getCharAt( end - 1 );
         for ( int32 tIdx=0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         end--;
      }

      // an empty string if set is empty
      vm->retval( cs->subString( start, end ) );
   }
}

/*#
   @function strFrontTrim
   @brief Removes white spaces from the front of the string.
   @param string The string to be trimmed.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed substring.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the beginning of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.
*/
FALCON_FUNC  mth_strFrontTrim ( ::Falcon::VMachine *vm )
{
   String *self;
   Item *trimChars;

   if ( vm->self().isMethodic() )
   {
      self = vm->self().asString();
      trimChars = vm->param(0);
   }
   else
   {
      Item *i_str = vm->param( 0 );
      if ( i_str == 0 || ! i_str->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      self = i_str->asString();
      trimChars = vm->param(1);
   }

   if (trimChars == 0 ) {
      CoreString *cs = new CoreString( *self );
      cs->frontTrim();
      vm->retval( cs );
   }
   else if ( ! trimChars->isString() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   else {
      int pos = 0;
      int32 len = self->length();
      String *trim = trimChars->asString();
      int32 tLen = trim->length();

      while( pos < len )
      {
         uint32 chr = self->getCharAt( pos );
         int found = 0;

         for ( int32 tIdx = 0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         pos++;
      }

      // has something to be trimmed?
      if ( pos < len )
         vm->retval( new CoreString( self->subString( pos, len ) ) );
      else
         vm->retval( new CoreString );
   }
}

/*#
   @function strBackTrim
   @brief Removes white spaces at both the beginning and the end of the string.
   @param string The string to be trimmed.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed substring.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the beginning and at the end of the string removed, is returned.
   If @b trimSet is not supplied, it defaults to space, tabulation characters,
   new lines and carriage returns. The original string is unmodified.
*/
FALCON_FUNC  mth_strBackTrim ( ::Falcon::VMachine *vm )
{
   String *self;
   Item *trimChars;

   if ( vm->self().isMethodic() )
   {
      self = vm->self().asString();
      trimChars = vm->param(0);
   }
   else
   {
      Item *i_str = vm->param( 0 );
      if ( i_str == 0 || ! i_str->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      self = i_str->asString();
      trimChars = vm->param(1);
   }

   if ( trimChars == 0 ) {
      CoreString *cs = new CoreString( *self );
      cs->backTrim();
      vm->retval( cs );
   }
   else if ( ! trimChars->isString() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   else
   {
      int32 pos = self->length()-1;
      String *trim = trimChars->asString();
      int32 tLen = trim->length();

      while ( pos >= 0 )
      {
         uint32 chr = self->getCharAt( pos );
         int found = 0;

         for ( int32 tIdx=0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         pos--;
      }

      // has something to be trimmed?
      if ( pos >= 0)
         vm->retval( new CoreString( self->subString( 0, pos + 1 ) ) );
      else
         vm->retval( new CoreString );
   }
}

/*#
   @function strReplace
   @brief Replaces the all the occurrences of a substring with another one.
   @param string The string where the replace will take place.
   @param substr The substring that will be replaced.
   @param repstr The string that will take the place of substr.
   @optparam start Optional start position in the string.
   @optparam end Optional end position in the string.
   @return A copy of the string with the occourences of the searched substring replaced.

   This is a flexible function that allows to alter a string by changing all the
   occurrences of a substring into another one. If the start parameter is given,
   the search and replacement will take place only starting at the specified
   position up to the end of the string, and if the end parameter is also provided,
   the search will take place in the interval [start, end-1].
*/

/*#
   @method replace String
   @brief Replaces the all the occurrences of a substring with another one.
   @param substr The substring that will be replaced.
   @param repstr The string that will take the place of substr.
   @optparam start Optional start position in the string.
   @optparam end Optional end position in the string.
   @return A copy of the string with the occourences of the searched substring replaced.

   @see strReplace
*/
FALCON_FUNC  mth_strReplace ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target;
   Item *needle;
   Item *replacer;
   Item *start_item;
   Item *end_item;

   if( vm->self().isMethodic() )
   {
      target = &vm->self();
      needle = vm->param(0);
      replacer = vm->param(1);
      start_item = vm->param(2);
      end_item = vm->param(3);
   }
   else
   {
      target = vm->param(0);
      needle = vm->param(1);
      replacer = vm->param(2);
      start_item = vm->param(3);
      end_item = vm->param(4);
   }

   if ( target == 0 || ! target->isString()
        || needle == 0 || ! needle->isString()
        || replacer == 0 || ! replacer->isString()
        || (start_item != 0 && ! start_item->isOrdinal())
        || (end_item != 0 && ! end_item->isOrdinal())
        )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "S,S,[N],[N]" : "S,S,S,[N],[N]"  ) );
   }

   // Parameter estraction.
   String *tg_str = target->asString();
   uint32 tg_len = target->asString()->length();

   String *ned_str = needle->asString();
   int32 ned_len = (int32) needle->asString()->length();

   // Is the needle is empty
   if ( ned_len == 0 ) {
      // shallow copy the target
      vm->retval( *target );
      return;
   }

   String *rep_str = replacer->asString();
   uint32 rep_len = replacer->asString()->length();

   int32 start = start_item ? (int32) start_item->forceInteger(): 0;
   if ( start < 0 ) start = 0;
   int32 end = end_item ? (int32) end_item->forceInteger(): tg_len-1;
   if ( end >= (int32) tg_len ) end = tg_len-1;

   CoreString *ret = new CoreString;
   if ( start > 0 )
      ret->append( String( *tg_str, 0, start ) );
   int32 old_start = start;
   while ( start <= end )
   {
      int32 ned_pos = 0;
      int32 pos = 0;
      // skip matching pattern
      while ( tg_str->getCharAt( start + pos ) == ned_str->getCharAt( ned_pos )
               && ned_pos < (int32) ned_len && start + ned_pos <= end )
      {
         ned_pos++;
         pos++;
      }

      // a match?
      if ( ned_pos == ned_len )
      {
         if ( start > old_start ) {
            ret->append( String( *tg_str, old_start, start ) );
         }

         if ( rep_len > 0 ) {
            ret->append( *rep_str );
         }

         start += ned_len;
         old_start = start;
      }
      else
         start++;
   }

   if ( old_start < (int32)tg_len )
      ret->append( String( *tg_str, old_start ) );

   vm->retval( ret );
}

/*#
   @function strReplicate
   @brief Returns a new string that is created by replicating the original one.
   @param string The string to be replicaeted.
   @param times Number of times the string must be replicated.
   @return The new string.

   A nice shortcut. Also, this function performs the work efficiently,
   preallocating the needed space in one shot and avoiding the need
   to grow the string while replicating the original value.
*/

/*#
   @method replicate String
   @brief Returns a new string that is created by replicating this one.
   @param times Number of times the string must be replicated.
   @return The new string.

   A nice shortcut. Also, this function performs the work efficiently,
   preallocating the needed space in one shot and avoiding the need
   to grow the string while replicating the original value.
*/
FALCON_FUNC  mth_strReplicate ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *strrep;
   Item *qty;

   if ( vm->self().isMethodic() )
   {
      strrep = &vm->self();
      qty = vm->param(0);
   }
   else
   {
      strrep = vm->param(0);
      qty = vm->param(1);
   }

   if ( strrep == 0 || ! strrep->isString()
        || qty == 0 || ! qty->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "N" : "S,N" ) );
   }

   int32 repl = (int32) qty->forceInteger();
   String *replicated = strrep->asString();
   int32 len = replicated->size() * repl;
   if ( len <= 0 ) {
      vm->retval( new CoreString("") );
      return;
   }

   CoreString *target = new CoreString;
   target->reserve( len );

   int pos = 0;
   while ( pos < len ) {
      memcpy( target->getRawStorage() + pos, replicated->getRawStorage(), replicated->size() );
      pos+= replicated->size();
   }

   target->manipulator( const_cast<Falcon::csh::Base*>(replicated->manipulator()->bufferedManipulator()) );
   target->size( len );
   vm->retval( target );
}

/*#
   @function strBuffer
   @brief Pre-allocates an empty string.
   @param size Size of the pre-allocated string.
   @return The new string.

   The returned string is an empty string, and equals to "". However, the required
   size is pre-allocated, and addition to this string (i.e. += operators)
   takes place in a fraction of the time otherwise required, up tho the filling
   of the pre-allocated buffer. Also, this string can be fed into file functions,
   the pre-allocation size being used as the input read size.
*/
FALCON_FUNC  strBuffer ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *qty = vm->param(0);
   if ( qty == 0 || ! qty->isOrdinal() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra("N") );
   }

   int32 size = (int32) qty->forceInteger();
   if ( size <= 0 ) {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ) );
   }

   vm->retval( new CoreString( String( size ) ) );
}

/*#
   @function strUpper
   @brief Returns an upper case version of the string.
   @param string String that must be uppercased.
   @return The uppercased string.

   All the Latin characters in the string are turned uppercase. Other characters
   are left untouched.
*/

/*#
   @method upper String
   @brief Returns an upper case version of this string.
   @return The uppercased string.

   All the Latin characters in the string are turned uppercase. Other characters
   are left untouched.
*/
FALCON_FUNC  mth_strUpper ( ::Falcon::VMachine *vm )
{
   Item *source;

   // Parameter checking;
   if ( vm->self().isMethodic() )
   {
      source = &vm->self();
   }
   else
   {
      source = vm->param(0);
      if ( source == 0 || ! source->isString() ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "S" ) );
      }
   }

   String *src = source->asString();
   if ( src->size() == 0 )
   {
      vm->retval( new CoreString );
   }
   else {
      CoreString *target = new CoreString( *src );
      target->upper();
      vm->retval( target );
   }
}

/*#
   @function strLower
   @brief Returns a lowercase version of the given string.
   @param string String that must be lowercased.
   @return The lowercased string.

   All the Latin characters in the string are turned lowercase. Other characters
   are left untouched.
*/

/*#
   @method lower String
   @brief Returns a lowercase version of this string.
   @return The lowercased string.

   All the Latin characters in the string are turned lowercase. Other characters
   are left untouched.
*/
FALCON_FUNC  mth_strLower ( ::Falcon::VMachine *vm )
{
   Item *source;

   // Parameter checking;
   if ( vm->self().isMethodic() )
   {
      source = &vm->self();
   }
   else
   {
      source = vm->param(0);
      if ( source == 0 || ! source->isString() ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "S" ) );
      }
   }

   String *src = source->asString();
   if ( src->size() == 0 )
   {
      vm->retval( new CoreString );
   }
   else {
      CoreString *target = new CoreString(  *src );
      target->lower();
      vm->retval( target );
   }
}


/*#
   @method startsWith String
   @brief Check if a strings starts with a substring.
   @param token The substring that will be compared with this string.
   @optparam icase If true, pefroms a case neutral check
   @return True if @b token matches the beginning of this string, false otherwise.

   This method performs a comparation check at the beginning of the string.
   If this string starts with @b token, the function returns true. If @b token
   is larger than the string, the method will always return false, and
   if @b token is an empty string, it will always match.

   The optional parameter @b icase can be provided as true to have this
   method to perform a case insensitive match.
*/
/*#
   @function strStartsWith
   @brief Check if a strings starts with a substring.
   @param string The string that is going to be tested for the given token.
   @param token The substring that will be compared with this string.
   @optparam icase If true, pefroms a case neutral check
   @return True if @b token matches the beginning of @b string, false otherwise.

   This functioin performs a comparation check at the beginning of the @b string.
   If this string starts with @b token, the function returns true. If @b token
   is larger than the string, the function will always return false, and
   if @b token is an empty string, it will always match.

   The optional parameter @b icase can be provided as true to have this
   function to perform a case insensitive match.
*/

FALCON_FUNC  mth_strStartsWith ( ::Falcon::VMachine *vm )
{
   Item* source;
   Item* i_token;
   Item* i_icase;

   // Parameter checking;
   if ( vm->self().isMethodic() )
   {
      source = &vm->self();
      i_token = vm->param(0);
      i_icase = vm->param(1);
   }
   else
   {
      source = vm->param(0);
      i_token = vm->param(1);
      i_icase = vm->param(2);
   }

   if ( source == 0 || ! source->isString() ||
        i_token == 0 || ! i_token->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "S,[B]" : "S,S,[B]" ) );
   }

   String *src = source->asString();
   vm->regA().setBoolean( src->startsWith( *i_token->asString(), i_icase ? i_icase->isTrue():false ) );
}


/*#
   @method endsWith String
   @brief Check if a strings ends with a substring.
   @param token The substring that will be compared with this string.
   @optparam icase If true, pefroms a case neutral check
   @return True if @b token matches the end of this string, false otherwise.

   This method performs a comparation check at the end of the string.
   If this string ends with @b token, the function returns true. If @b token
   is larger than the string, the method will always return false, and
   if @b token is an empty string, it will always match.

   The optional parameter @b icase can be provided as true to have this
   method to perform a case insensitive match.
*/
/*#
   @function strEndsWith
   @brief Check if a strings ends with a substring.
   @param string The string that is going to be tested for the given token.
   @param token The substring that will be compared with this string.
   @optparam icase If true, pefroms a case neutral check
   @return True if @b token matches the end of @b string, false otherwise.

   This functioin performs a comparation check at the end of the @b string.
   If this string ends with @b token, the function returns true. If @b token
   is larger than the string, the function will always return false, and
   if @b token is an empty string, it will always match.

   The optional parameter @b icase can be provided as true to have this
   function to perform a case insensitive match.
*/
FALCON_FUNC  mth_strEndsWith ( ::Falcon::VMachine *vm )
{
   Item* source;
   Item* i_token;
   Item* i_icase;

   // Parameter checking;
   if ( vm->self().isMethodic() )
   {
      source = &vm->self();
      i_token = vm->param(0);
      i_icase = vm->param(1);
   }
   else
   {
      source = vm->param(0);
      i_token = vm->param(1);
      i_icase = vm->param(2);
   }

   if ( source == 0 || ! source->isString() ||
        i_token == 0 || ! i_token->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "S,[B]" : "S,S,[B]" ) );
   }

   String *src = source->asString();
   vm->regA().setBoolean( src->endsWith( *i_token->asString(), i_icase ? i_icase->isTrue() : false ) );
}


/*#
   @function strCmpIgnoreCase
   @brief Performs a lexicographic comparation of two strings, ignoring character case.
   @param string1 First string to be compared.
   @param string2 Second string to be compared.
   @return -1, 0 or 1.

   The two strings are compared ignoring the case of latin characters contained in
   the strings.

   If the first string is greater than the second, the function returns a number
   less than 0. If it's
   smaller, it returns a positive number. If the two strings are the same,
   ignoring the case of the characters, 0 is returned.

*/
/*#
   @method cmpi String
   @brief Performs a lexicographic comparation of two strings, ignoring character case.
   @param string Second string to be compared with this one.
   @return <0, 0 or >0, respectively if this string is less, equal or greater than
      the @b string parameter.

   The two strings are compared ignoring the case of latin characters contained in
   the strings.

   If the first string is greater than the second, the function returns a number
   less than 0. If it's
   smaller, it returns a positive number. If the two strings are the same,
   ignoring the case of the characters, 0 is returned.
*/
FALCON_FUNC  mth_strCmpIgnoreCase ( ::Falcon::VMachine *vm )
{
   Item *s1_itm;
   Item *s2_itm;

   // Parameter checking;
   if( vm->self().isMethodic() )
   {
      s1_itm = &vm->self();
      s2_itm = vm->param(0);
   }
   else
   {
      s1_itm = vm->param(0);
      s2_itm = vm->param(1);
   }

   if ( s1_itm == 0 || ! s1_itm->isString()
       || s2_itm == 0 || !s2_itm->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "S" : "S,S" ) );
   }

   String *str1 = s1_itm->asString();
   String *str2 = s2_itm->asString();

   int32 len1 = str1->length();
   int32 len2 = str2->length();

   int32 minlen = len1 > len2 ? len2 : len1;

   for( int32 i = 0; i < minlen; i ++ )
   {
      int32 elem1 = str1->getCharAt( i );
      int32 elem2 = str2->getCharAt( i );
      if ( elem1 >= 'A' && elem1 <= 'Z' )
         elem1 |= 0x20;
      if ( elem2 >= 'A' && elem2 <= 'Z' )
         elem2 |= 0x20;

      if ( elem1 > elem2 ) {
         vm->retval( 1 );
         return;
      }

      if ( elem1 < elem2 ) {
         vm->retval( -1 );
         return;
      }
   }

   if ( len1 > len2 ) {
      vm->retval( 1 );
      return;
   }

   if ( len1 < len2 ) {
      vm->retval( -1 );
      return;
   }

   // same!!!
   vm->retval( 0 );
}


inline void internal_escape( int mode, VMachine* vm )
{
   Item *i_string;
   Item *i_onplace;

   // Parameter checking;
   if( vm->self().isMethodic() )
   {
      i_string = &vm->self();
      i_onplace = vm->param(0);
   }
   else
   {
      i_string = vm->param(0);
      i_onplace = vm->param(1);
      if ( i_string == 0 || ! i_string->isString() )
         {
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .origin( e_orig_runtime )
                  .extra( "S,B" ) );
         }
   }

   String* str;

   if ( mode != 2 && i_onplace != 0 && i_onplace->isTrue() )
   {
      str = i_string->asString();

   }
   else
   {
      str = new CoreString( *i_string->asString() );
   }

   switch( mode )
   {
   case 0: //esq
      str->escapeQuotes();
      break;

   case 1: //unesq
      str->unescapeQuotes();
      break;

   case 2: //escape
   {
      CoreString* other = new CoreString( str->size() );
      if ( i_onplace != 0 && i_onplace->isTrue() ) // actually it means "complete"
         str->escapeFull( *other );
      else
         str->escape( *other );

      vm->retval( other );
   }
   return;


   case 3: //unescape
      str->unescape();
      break;
   }

   vm->retval( str );
}

/*#
   @function strEsq
   @brief Escape quotes in the given string.
   @param string the String to be escaped.
   @optparam inplace if true, the source string is modified, saving memory.
   @return A new escaped string, if @b inplace is not given, or the @b string parameter
           if @b inplace is true.


   @see String.esq
   @see strUnesq
*/
/*#
   @method esq String
   @brief Escapes the quotes in this string.
   @optparam inplace if true, the source string is modified, saving memory.
   @return A new escaped string, if @b inplace is not given, or this same string
           if @b inplace is true.

   @see String.unesq
   @see strEsq
*/

FALCON_FUNC  mth_strEsq ( ::Falcon::VMachine *vm )
{
   internal_escape( 0, vm );
}

/*#
   @function strUnesq
   @brief Unescape the quotes in given string.
   @param string the String to be unescaped.
   @optparam inplace if true, the source string is modified, saving memory.
   @return A new unescaped string, if @b inplace is not given, or the @b string parameter
           if @b inplace is true.

   This function transforms all the occourences of '\\"' and '\\'' into a double or
   single quote, leaving all the other special escape sequences untouched.

   @see String.unesq
   @see strEsq
*/
/*#
   @method unesq String
   @brief Escapes the quotes in this string.
   @optparam inplace if true, the source string is modified, saving memory.
   @return A new escaped string, if @b inplace is not given, or this same string
           if @b inplace is true.

   @see String.esq
   @see strUnesq
*/

FALCON_FUNC  mth_strUnesq ( ::Falcon::VMachine *vm )
{
   internal_escape( 1, vm );
}


/*#
   @function strEscape
   @brief Escape quotes and special characters in the string
   @param string the String to be escaped.
   @optparam full If true, characters above UNICODE 127 are escaped as well.
   @return A new escaped string.


   @see String.esq
   @see strUnesq
*/
/*#
   @method escape String
   @brief Escapes all the special characters in the string.
   @optparam full If true, characters above UNICODE 127 are escaped as well.
   @return A new escaped string.

   @see String.esq
   @see strEsq
   @see strUnescape
*/

FALCON_FUNC  mth_strEscape ( ::Falcon::VMachine *vm )
{
   internal_escape( 2, vm );
}

/*#
   @function strUnescape
   @brief Unescape quotes and special characters in the string
   @param string the String to be escaped.
   @optparam inplace if true, the source string is modified, saving memory.
   @return A new unescaped string, if @b inplace is not given, or the @b string parameter
           if @b inplace is true.



   @see String.esq
   @see strUnesq
   @see String.unescape
*/
/*#
   @method unescape String
   @brief Unescapes all the special characters in the string.
   @optparam inplace if true, the source string is modified, saving memory.
   @return A new unescaped string, if @b inplace is not given, or the @b string parameter
           if @b inplace is true.


   @see String.esq
   @see strEsq
   @see strEscape
*/

FALCON_FUNC  mth_strUnescape ( ::Falcon::VMachine *vm )
{
   internal_escape( 3, vm );
}

/*#
   @function strWildcardMatch
   @brief Perform an old-style file-like jolly-based wildcard match.
   @param string String that must match the wildcard.
   @param wildcard A wildcard, possibly but not necessarily including a jolly character.
   @optparam ignoreCase If true, the latin 26 base letters case is ignored in matches.
   @return True if the string matches, false otherwise.

   This function matches a wildcard that may contain jolly "*" or "?" characters against a
   string, eventually ignoring the case of the characters. This is a practical function
   to match file names against given patterns. A "?" in the wildcard represents any
   single character, while a "*" represents an arbitrary sequence of characters.

   The wildcard must match completely the given string for the function to return true.

   For example:
   - "*" matches everything
   - "a?b" matches "aab", "adb" and so on
   - "a*b" matches "ab", "annnb" and so on

   @see String.wmatch
*/

/*#
   @method wmatch String
   @brief Perform an old-style file-like jolly-based wildcard match.
   @param wildcard A wildcard, possibly but not necessarily including a jolly character.
   @optparam ignoreCase If true, the latin 26 base letters case is ignored in matches.
   @return True if the string matches, false otherwise.

   This function matches a wildcard that may contain jolly "*" or "?" characters against a
   string, eventually ignoring the case of the characters. This is a practical function
   to match file names against given patterns. A "?" in the wildcard represents any
   single character, while a "*" represents an arbitrary sequence of characters.

   The wildcard must match completely the given string for the function to return true.

   For example:
   - "*" matches everything
   - "a?b" matches "aab", "adb" and so on
   - "a*b" matches "ab", "annnb" and so on

   @see strWildcardMatch
*/
FALCON_FUNC  mth_strWildcardMatch ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *s1_itm = vm->param(0);
   Item *s2_itm = vm->param(1);
   Item *i_bIcase = vm->param(2);

   if ( vm->self().isMethodic() )
   {
      s1_itm = &vm->self();
      s2_itm = vm->param(0);
      i_bIcase = vm->param(1);
   }
   else
   {
      s1_itm = vm->param(0);
      s2_itm = vm->param(1);
      i_bIcase = vm->param(2);
   }

   if ( s1_itm == 0 || ! s1_itm->isString() || s2_itm == 0 || !s2_itm->isString() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
           .origin( e_orig_runtime )
           .extra( vm->self().isMethodic() ? "S,[B]" : "S,S,[B]") );
   }

   // Ignore case?
   bool bIcase = i_bIcase == 0 ? false : i_bIcase->isTrue();

   // The first is the wildcard, the second is the matched thing.
   String *cfr = s1_itm->asString();
   String *wcard = s2_itm->asString();

   vm->regA().setBoolean( cfr->wildcardMatch( *wcard, bIcase ) );
}

/*#
   @function strToMemBuf
   @brief Convets a string into a Memory Buffer
   @param string String to be converted in a membuf.
   @optparam wordWidth The memory buffer word width (defaults to string character size).
   @return The resulting membuf.

   This function creates a membuf from a string. The resulting membuf
   has the same word width of the original string, which may be 1, 2 or 4
   byte wide depending on the size needed to store its contents. It is possible
   to specify a different word width; in that case the function will be much
   less efficient (each character must be copied).

   If wordWidth is set to zero, the resulting memory buffer will have 1 byte
   long elements, but the content of the string will be copied as-is, bytewise,
   regardless of its character size.
*/

/*#
   @method toMemBuf String
   @brief Convets this string into a Memory Buffer
   @optparam wordWidth The memory buffer word width (defaults to string character size).
   @return The resulting membuf.

   This function creates a membuf from a string. The resulting membuf
   has the same word width of the original string, which may be 1, 2 or 4
   byte wide depending on the size needed to store its contents. It is possible
   to specify a different word width; in that case the function will be much
   less efficient (each character must be copied).

   If wordWidth is set to zero, the resulting memory buffer will have 1 byte
   long elements, but the content of the string will be copied as-is, bytewise,
   regardless of its character size.
*/


FALCON_FUNC  mth_strToMemBuf ( ::Falcon::VMachine *vm )
{
   Item *i_string;
   Item *i_wordWidth;

   // Parameter checking;
   if ( vm->self().isMethodic() )
   {
      i_string = &vm->self();
      i_wordWidth = vm->param(0);
   }
   else
   {
      i_string = vm->param(0);
      i_wordWidth = vm->param(1);
   }

   if( i_string == 0 || ! i_string->isString()
      || ( i_wordWidth != 0 && ! i_wordWidth->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "[N]" : "S,[N]" ) );
   }

   String *string = i_string->asString();
   int charSize = string->manipulator()->charSize();
   int64 ww = i_wordWidth == 0 ? charSize : i_wordWidth->forceInteger();
   MemBuf *result;

   if ( ww == 0 )
   {
      result = new MemBuf_1( string->size() );
      memcpy( result->data(), string->getRawStorage(), string->size() );
   }
   else
   {
      result = MemBuf::create( charSize, string->length() );

      if ( result == 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
           .origin( e_orig_runtime )
           .extra("0-4") );
      }

      if ( ww == charSize )
      {
         memcpy( result->data(), string->getRawStorage(), string->size() );
      }
      else
      {
         uint32 size = string->size();
         for( uint32 p = 0; p < size; p++ )
         {
            result->set( p, string->getCharAt(p) );
         }
      }
   }

   vm->retval( result );
}

/*#
   @function strFromMemBuf
   @brief Convets a MemBuf to a string.
   @param membuf A MemBuf that will be converted to a string.
   @return The resulting string.

   This string takes each element of the membuf and converts it into
   a character in the final string. The contents of the buffer are
   not transcoded. It is appropriate to say that this function considers
   each element in the MemBuf as an Unicode value for the character in the
   final string.

   To create a string from a buffer that may come from an encoded source
   (i.e. a file), use directly Transcode functions.
*/

FALCON_FUNC  strFromMemBuf ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *i_membuf = vm->param(0);

   if( i_membuf == 0 || ! i_membuf->isMemBuf() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra("M") );
   }

   MemBuf *mb = i_membuf->asMemBuf();

   // preallocating size instead of len, we won't have to resize the memory even
   // if resizing character sizes.
   CoreString *result = new CoreString(  mb->size() );

   uint32 size = mb->length();
   for( uint32 p = 0; p < size; p++ )
   {
      result->append( mb->get( p ) );
   }

   vm->retval( result );
}


/*#
   @function strFill
   @brief Fills a string with a given character or substring.
   @param string The string to be filled.
   @param chr The character (unicode value) or substring used to refill the @b string.
   @return The string itself.

   This function fills the physical storage of the given string with a single
   character or a repeated substring. This can be useful to clean a string used repeatedly
   as input buffer.

   The function returns the same string that has been passed as the parameter.
*/

/*#
   @method fill String
   @brief Fills a string with a given character or substring.
   @param chr The character (unicode value) or substring used to refill this string.
   @return The string itself.

   This method fills the physical storage of the given string with a single
   character or a repeated substring. This can be useful to clean a string used repeatedly
   as input buffer.
*/


FALCON_FUNC  mth_strFill ( ::Falcon::VMachine *vm )
{
   Item *i_string;
   Item *i_chr;

   // Parameter checking;
   if ( vm->self().isMethodic() )
   {
      i_string = &vm->self();
      i_chr = vm->param(0);
   }
   else
   {
      i_string = vm->param(0);
      i_chr = vm->param(1);
   }

   if( i_string == 0 || ! i_string->isString()
      || i_chr == 0 || ( ! i_chr->isOrdinal() && !i_chr->isString())
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "N|S" : "S,N|S" ) );
   }

   CoreString *string = i_string->asCoreString();

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
            .origin( e_orig_runtime )
            .extra( vm->moduleString( rtl_string_empty ) ) );
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

   vm->retval( string );
}

}}


/* end of string.cpp */
