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
#include <falcon/stderrors.h>

#include <falcon/itemarray.h>
#include <falcon/function.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

namespace Falcon {

/**
 @class String
 @optparam item

 @prop len Length of the string (in characters)
 @prop allocated Size of the memory occupied by the strings
 @prop isText If true, the string is text-oriented, otherwise it's a memory buffer.
 @prop mutable True if he string is mutable.
 @prop allAlpha (static) Returns an immutable string containing all the upper and lower case latin letters.
 @prop allAlphaNum (static) Returns an immutable string containing all the upper and lower case latin letters and digits.
 @prop allDigit (static) Returns an immutable string containing all the digits.
 @prop allUpper (static) Returns an immutable string containing all the upper case latin letters.
 @prop allLower (static) Returns an immutable string containing all the lower case latin letters.
 @prop allPunct (static) Returns an immutable string containing all the common puntaction characters.

 @brief Language level class handling string data type.

 @section string_mutable Mutable strings

 @section string_re Regular expression strings

 @section string_i International strings

 @section string_ops Standard operators
 - add (+): String + <anything> => concatenates this item and the string
             representation of the other item
 - aadd (+=): String += <anything> => concatenates this item in place;
             the original string is changed and lengthened (only if mutable).
 - mul (*): String * <number> => Creates a new string containing <number> copies of
              the original string.
 - amul (*=): String *= <number> => Takes the string and concatenaes the given number
              of copies to it (only if mutable).
 - div (/): String / <number> => Generates a character having the UNICODE value of the
           last character in the string plus the numeric value. For instance, "B"/1 gives "C",
           and "B"/-1 gives "A".
 - mod (%): String / <number> => Concatenates the given string with the UNICODE character
           represented by the number. Equivalent to "..." + "\x<number>"
 - amod (%=): String /= <number> => Concatenates the given string with the UNICODE character
           represented by the number, appending it directly in place after the string.
           Equivalent to "..." += "\x<number>" (only if mutable)

 - index ([]): The operator gives access to the nth character in the string, or to a substring
    range in case the index is a range. Setting new values as single characters or ranges
    is possible if the string is mutable only.
 - star-index ([*n]): Returns the UNICODE value of the nth character, or if the string is non-text,
    it's byte value.

  - in: String1 in String2: true if string1 is a substring (is found in) string2
     Equivalent to string2.find(string1) >= 0.

  @section string_iter Iteration over strings

  Strings are iterable; this means they can be used as expressions in for-in
  loops, as terms for the accumulator expression ^[], and the @a BOM.foreach
  method can iterate over them.

  Each iteration results in a mutable string containing exactly one character,
  so that

  @code
  "hello".foreach(printl)
  @endcode

  prints each character of hello on a single line.
 */
//===============================================================================
// Opcodes
//

class ClassString::PStepInitNext: public PStep
{
public:
   PStepInitNext() { apply = apply_; }
   virtual ~PStepInitNext() {}
   virtual void describeTo(String& tg) const { tg = "ClassString::PStepInitNext";}
   static void apply_( const PStep*, VMContext* vm );
};


void ClassString::PStepInitNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->opcodeParam(1).asString()->copy( *ctx->topData().asString() );
   // remove the locally pushed data and the parameters.
   ctx->popData( 2 + ctx->currentCode().m_seqId );
   ctx->popCode();
}


class ClassString::PStepNextOp: public PStep
{
public:
   PStepNextOp( ClassString* owner ): m_owner(owner) { apply = apply_; }
   virtual ~PStepNextOp() {}
   virtual void describeTo(String& tg) const { tg = "ClassString::PStepNextOp";}
   static void apply_( const PStep*, VMContext* vm );

private:
   ClassString* m_owner;
};


void ClassString::PStepNextOp::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepNextOp* step = static_cast<const PStepNextOp*>(ps);

   // The result of a deep call is in A
   Item* op1, *op2;

   ctx->operands( op1, op2 ); // we'll discard op2

   String* deep = op2->asString();
   String* self = op1->asString();

   ctx->popData();
   InstanceLock::Token* tk = step->m_owner->m_lock.lock(self);
   self->append( *deep );
   step->m_owner->m_lock.unlock(tk);
   ctx->popCode();
}


//=====================================================================
// Properties
//@b count

static void get_len( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<String*>( instance )->length();
}

static void get_mutable( const Class*, const String&, void* instance, Item& value )
{
   value.setBoolean(! static_cast<String*>( instance )->isImmutable() );
}


static void get_allocated( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<String*>( instance )->allocated();
}

static void get_isText( const Class*, const String&, void* instance, Item& value )
{
   value.setBoolean( static_cast<String*>( instance )->isText() );
}

static void set_isText( const Class* cls, const String&, void* instance, const Item& value )
{
   String* str = static_cast<String*>( instance );
   const ClassString* cstring = static_cast<const ClassString*>(cls);

   if( str->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("Immutable string") );
   }

   InstanceLock::Token* tk = cstring->lockInstance(str);
   if( value.isTrue() ) {
      if( ! str->isText() ) {
         str->manipulator( str->manipulator()->bufferedManipulator() );
      }
   }
   else {
      str->toMemBuf();
   }
   cstring->unlockInstance(tk);
}

/*#
   @property charSize String
   @brief Returns or changes the size in bytes in this string.

   This properties control the count bytes per character (1, 2 or 4) used to store
   each character in this string.
*/


static void get_charSize( const Class*, const String&, void* instance, Item& value )
{
   String* str = static_cast<String*>(instance);
   value.setInteger( str->manipulator()->charSize() );
}

static void set_charSize( const Class* cls, const String&, void* instance, const Item& value )
{
   String* str = static_cast<String*>(instance);

   if( ! value.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_inv_params, __LINE__, SRC )
         .extra( "N" ) );
   }

   uint32 bpc = (uint32) value.isOrdinal();
   const ClassString* cstring = static_cast<const ClassString*>(cls);
   InstanceLock::Token* tk = cstring->lockInstance(str);
   if ( ! str->setCharSize( bpc ) )
   {
      throw new  OperandError( ErrorParam( e_param_range, __LINE__, SRC ) );
   }
   cstring->unlockInstance(tk);
}


static void get_allAlpha( const Class* cls, const String&, void*, Item& value )
{
   const ClassString* scls = static_cast<const ClassString*>(cls);
   value.setUser(scls, scls->m_modelAllAlpha );
}

static void get_allAlphaNum( const Class* cls, const String&, void*, Item& value )
{
   const ClassString* scls = static_cast<const ClassString*>(cls);
   value.setUser(scls, scls->m_modelAllAlphaNum );
}

static void get_allDigit( const Class* cls, const String&, void*, Item& value )
{
   const ClassString* scls = static_cast<const ClassString*>(cls);
   value.setUser(scls, scls->m_modelAllDigit );
}

static void get_allUpper( const Class* cls, const String&, void*, Item& value )
{
   const ClassString* scls = static_cast<const ClassString*>(cls);
   value.setUser(scls, scls->m_modelAllUpper );
}

static void get_allLower( const Class* cls, const String&, void*, Item& value )
{
   const ClassString* scls = static_cast<const ClassString*>(cls);
   value.setUser(scls, scls->m_modelAllLower );
}

static void get_allPunct( const Class* cls, const String&, void*, Item& value )
{
   const ClassString* scls = static_cast<const ClassString*>(cls);
   value.setUser(scls, scls->m_modelAllPunct );
}

//=========================================================================
//
//

namespace _classString
{

/*#
   @method front String
   @brief Returns the first characters in a string.
   @param count Number of characters to be returned.
   @return The first elements or an empty string if the string is empty

   This method returns a string containing characters from the beginning of the string.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( front, "string:S,count:N" );
FALCON_DEFINE_FUNCTION_P1(front)
{
   String *str;
   Item* i_len;
   int64 len;

   if( ctx->isMethodic() )
   {
      str = ctx->self().asString();
      i_len = ctx->param(0);
      if( ! i_len->isOrdinal() ) throw paramError( __LINE__, SRC, true );
      len = i_len->forceInteger();
   }
   else {
      Item* i_str = ctx->param(0);
      i_len = ctx->param(1);

      if( ! i_str->isString() || ! i_len->isOrdinal() ) {
         throw paramError( __LINE__, SRC, true );
      }

      str = i_str->asString();
      len = i_len->forceInteger();
   }

   ClassString* cstring = static_cast<ClassString*>(methodOf());
   Item result;
   InstanceLock::Token* tk = cstring->lockInstance(str);
   if ( len <= 0 || str->length() == 0 )
   {
      result = FALCON_GC_HANDLE(new String);
   }
   else if ( ((length_t) len) >= str->length() ) {
      result = FALCON_GC_HANDLE(new String( *str ));
   }
   else {
      result = FALCON_GC_HANDLE(new String( *str, 0, static_cast<length_t>(len) ));
   }
   cstring->unlockInstance(tk);

   ctx->returnFrame( result );
}

/*#
   @method back String
   @brief Returns the last characters in a string.
   @param count Number of characters to be returned.
   @return The last elements or an empty string if the string is empty

   This method returns a string containing characters from the end of the string.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( back, "back:S,count:N" );
FALCON_DEFINE_FUNCTION_P1(back)
{
   String *str;
   Item* i_len;
   int64 len;

   if( ctx->isMethodic() )
   {
      str = ctx->self().asString();
      i_len = ctx->param(0);
      if( ! i_len->isOrdinal() ) throw paramError( __LINE__, SRC, true );
      len = i_len->forceInteger();
   }
   else {
      Item* i_str = ctx->param(0);
      i_len = ctx->param(1);

      if( ! i_str->isString() || ! i_len->isOrdinal() ) {
         throw paramError( __LINE__, SRC, true );
      }

      str = i_str->asString();
      len = i_len->forceInteger();
   }

   ClassString* cstring = static_cast<ClassString*>(methodOf());
   Item result;
   InstanceLock::Token* tk = cstring->lockInstance(str);
   if ( len <= 0 || str->length() == 0 )
   {
      result = FALCON_GC_HANDLE(new String);
   }
   else if ( ((length_t) len) >= str->length() ) {
      result = FALCON_GC_HANDLE(new String( *str ));
   }
   else {
      result = FALCON_GC_HANDLE(new String( *str, static_cast<length_t>(str->length()-len) ));
   }
   cstring->unlockInstance(tk);

   ctx->returnFrame( result );
}



/*#
   @method splittr String
   @brief Subdivides a string in an array of substrings given a token substring.
   @optparam token The token by which the string should be split.
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

   Differently to @a String.split, this function will "eat up" adjacent tokens. While
   @a strSplit is more adequate to parse field-oriented strings (as i.e.
   colon separated fields in configuration files) this function is best employed
   in word extraction.

   @note See @a Tokenizer for a more adequate function to scan extensively
   wide strings.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( splittr, "token:[S],count:[N]" );
FALCON_DEFINE_FUNCTION_P1(splittr)
{
   Item *target;
   Item *splitstr;
   Item *count;

   // Parameter checking;
   if( ctx->isMethodic() )
   {
      target = &ctx->self();
      splitstr = ctx->param(0);
      count = ctx->param(1);
   }
   else
   {
      target = ctx->param(0);
      splitstr = ctx->param(1);
      count = ctx->param(2);
   }

   uint32 limit;

   if ( target == 0 || ! target->isString()
        || (splitstr != 0 && ! (splitstr->isString() || splitstr->isNil()))
        || ( count != 0 && ! count->isOrdinal() ) )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   // Parameter extraction.
   limit = count == 0 ? 0xffffffff: (uint32) count->forceInteger();

   String *tg_str = target->asString();
   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk_str = cstring->lockInstance(tg_str);
   uint32 tg_len = target->asString()->length();

   // split in chars?
   if( splitstr == 0 || splitstr->isNil() || splitstr->asString()->size() == 0 )
   {
      // split the string in an array.
      if( limit > tg_len )
         limit = tg_len;

      ItemArray* ca = new ItemArray( limit );
      for( uint32 i = 0; i + 1 < limit; ++i )
      {
         String* elem = new String(1);
         elem->append( tg_str->getCharAt(i) );
         ca->append( FALCON_GC_HANDLE(elem) );
      }
      // add remains if there are any
      if(limit <= tg_len)
         ca->append(tg_str->subString(limit - 1));

      cstring->unlockInstance(tk_str);

      ctx->returnFrame(FALCON_GC_HANDLE(ca));
      return;
   }

   String* mstr = splitstr->asString();
   InstanceLock::Token* tk1 = cstring->lockInstance(mstr);
   String sp_str( *mstr );
   cstring->unlockInstance(tk1);

   uint32 sp_len = sp_str.length();

   // return item.
   ItemArray *retarr = new ItemArray;

   // if the split string is empty, return empty string
   if ( sp_len == 0 )
   {
      retarr->append( FALCON_GC_HANDLE(new String()) );
      cstring->unlockInstance(tk_str);
      ctx->returnFrame(FALCON_GC_HANDLE(retarr));
      return;
   }

   // if the token is wider than the string, just return the string
   if ( tg_len < sp_len )
   {
      retarr->append( FALCON_GC_HANDLE(new String( *tg_str )) );
      cstring->unlockInstance(tk_str);
      ctx->returnFrame(FALCON_GC_HANDLE(retarr));
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
      while ( tg_str->getCharAt( pos ) == sp_str.getCharAt( sp_pos ) &&
              sp_pos < sp_len && pos <= tg_len - sp_len ) {
         sp_pos++;
         pos++;
      }

      // a match?
      if ( sp_pos == sp_len ) {
         // put the item in the array.
         uint32 splitend = pos - sp_len;
         retarr->append( FALCON_GC_HANDLE(new String( String( *tg_str, last_pos, splitend ) ) ) );

         lastIsEmpty = (last_pos >= splitend);

         last_pos = pos;
         limit--;
         // skip matching pattern
         while( sp_pos == sp_len && pos <= tg_len - sp_len ) {
            sp_pos = 0;
            last_pos = pos;
            while ( tg_str->getCharAt( pos ) == sp_str.getCharAt( sp_pos )
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
      retarr->append( FALCON_GC_HANDLE(new String( String( *tg_str, last_pos, (uint32) tg_len ) ) ) );
   }

   cstring->unlockInstance(tk_str);
   ctx->returnFrame(FALCON_GC_HANDLE(retarr));
}


/*#
   @method split String
   @brief Subdivides a string in an array of substrings given a token substring.
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


   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( split, "string:S,token:[S],count:[N]" );
FALCON_DEFINE_FUNCTION_P1(split)

{
   Item *target;
   Item *splitstr;
   Item *count;

   // Parameter checking;
   if( ctx->isMethodic() )
   {
      target = &ctx->self();
      splitstr = ctx->param(0);
      count = ctx->param(1);
   }
   else
   {
      target = ctx->param(0);
      splitstr = ctx->param(1);
      count = ctx->param(2);
   }


   if ( (target == 0 || ! target->isString())
        || (splitstr != 0 && ! (splitstr->isString() || splitstr->isNil()))
        || ( count != 0 && ! count->isOrdinal() ) )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   // Parameter extraction.
   uint32 limit = count == 0 ? 0xffffffff: (uint32) count->forceInteger();

   String *tg_str = target->asString();
   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk_str = cstring->lockInstance(tg_str);
   uint32 tg_len = target->asString()->length();

   // split in chars?
   if( splitstr == 0 || splitstr->isNil() || splitstr->asString()->size() == 0)
   {
      // split the string in an array.
      if( limit > tg_len )
         limit = tg_len;

      ItemArray* ca = new ItemArray( limit );
      for( uint32 i = 0; i+1 < limit; ++i )
      {
         String* elem = new String(1);
         elem->append( tg_str->getCharAt(i) );
         ca->append( FALCON_GC_HANDLE(elem) );
      }
      // add remains if there are any
      if(limit <= tg_len)
         ca->append(tg_str->subString(limit - 1));

      cstring->unlockInstance(tk_str);

      ctx->returnFrame( FALCON_GC_HANDLE(ca) );
      return;
   }

   String* mstr = splitstr->asString();
   InstanceLock::Token* tk1 = cstring->lockInstance(mstr);
   String sp_str( *mstr );
   cstring->unlockInstance(tk1);

   uint32 sp_len = sp_str.length();

   // return item.
   ItemArray *retarr = new ItemArray;

   // if the split string is empty, return empty string
   if ( sp_len == 0 )
   {
      cstring->unlockInstance(tk_str);

      retarr->append( FALCON_GC_HANDLE(new String()) );
      ctx->returnFrame( FALCON_GC_HANDLE(retarr) );
      return;
   }

   // if the token is wider than the string, just return the string
   if ( tg_len < sp_len )
   {
      retarr->append( FALCON_GC_HANDLE(new String( *tg_str ) ) );
      cstring->unlockInstance(tk_str);

      ctx->returnFrame( FALCON_GC_HANDLE(retarr) );
      return;
   }

   uint32 pos = 0;
   uint32 last_pos = 0;
   // scan the string
   while( limit > 1 && pos <= tg_len - sp_len  )
   {
      uint32 sp_pos = 0;
      // skip matching pattern-
      while ( tg_str->getCharAt( pos ) == sp_str.getCharAt( sp_pos ) &&
              sp_pos < sp_len && pos <= tg_len - sp_len ) {
         sp_pos++;
         pos++;
      }

      // a match?
      if ( sp_pos == sp_len ) {
         // put the item in the array.
         uint32 splitend = pos - sp_len;
         retarr->append( FALCON_GC_HANDLE(new String( *tg_str, last_pos, splitend )) );
         last_pos = pos;
         limit--;

      }
      else
         pos++;
   }

   // Residual element?
   if ( limit >= 1 || last_pos < tg_len ) {
      uint32 splitend = tg_len;
      retarr->append( FALCON_GC_HANDLE(new String( *tg_str, last_pos, splitend )) );
   }

   cstring->unlockInstance(tk_str);
   ctx->returnFrame( FALCON_GC_HANDLE(retarr) );
}


/*#
   @method merge String
   @brief Merges an array of strings into one.
   @param array An array of strings to be merged.
   @optparam separator The separator used to merge the strings.
   @optparam count Maximum count of merges.
   @return The merged string.

   The method will return a new string containing the concatenation
   of the strings inside the array parameter. If the array is empty,
   an empty string is returned. If a mergeStr parameter is given, it
   is added to each pair of string merged; mergeStr is never added at
   the end of the new string. A count parameter may be specified to
   limit the number of elements merged in the array.

   The function may be used in this way:
   @code
   a = " ".merge( [ "a", "string", "of", "words" ] )
   printl( a ) // prints "a string of words"
   @endcode

   Or, using the function statically from the class:

   @code
   a = String.merge( " ", [ "a", "string", "of", "words" ] )
   printl( a ) // prints "a string of words"
   @endcode

   If an element of the array is not a string, an error is raised.

   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( merge, "separator:[S],array:[S],count:[N]" );
FALCON_DEFINE_FUNCTION_P1(merge)
{
   Item *mergestr, *source, *count;

   // Parameter checking;
   if( ctx->isMethodic() )
   {
      mergestr = &ctx->self();
      source = ctx->param(0);
      count = ctx->param(1);
   }
   else {
      mergestr = ctx->param(0);
      source = ctx->param(1);
      count = ctx->param(2);
   }
   uint64 limit;

   if ( source == 0 || ! source->isArray()
        || ( mergestr != 0 && ! mergestr->isString() )
        || ( count != 0 && ! count->isNil() && ! count->isOrdinal() ) )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic() );
   }

   // Parameter estraction.
   limit = count == 0 || count->isNil() ? 0xffffffff: count->forceInteger();

   String* mstr = mergestr->asString();
   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(mstr);
   String mr_str( *mstr );
   cstring->unlockInstance(tk1);

   ItemArray *elements = source->asArray();
   uint32 len = elements->length();
   if ( limit < len )
   {
      len = (uint32) limit;
   }

   String *ts = new String;

   // filling the target.
   for( uint32 i = 1; i <= len ; i ++ )
   {
      Item* item = &elements->at(i-1);

      if ( item->isString() )
      {
         String* str = item->asString();
         ClassString* cstring = static_cast<ClassString*>(methodOf());
         InstanceLock::Token* tk1 = cstring->lockInstance(str);
         *ts += *str;
         cstring->unlockInstance(tk1);
      }
      else
      {
         delete ts;
         throw FALCON_SIGN_XERROR( ParamError, e_param_type, .extra("Need to be string") );
      }

      if ( i < len )
         ts->append( mr_str );

      ts->reserve( len/i * ts->size() );
   }

   ctx->returnFrame( FALCON_GC_HANDLE(ts) );
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

   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( join, "separator:S,..." );
FALCON_DEFINE_FUNCTION_P1(join)
{
   Item *mergestr;
   int32 first;

   // Parameter checking;
   if( ctx->isMethodic() )
   {
      mergestr = &ctx->self();
      first = 0;
   }
   else {
      mergestr = ctx->param(0);
      first = 1;
   }

   if ( mergestr != 0 && ! mergestr->isString() )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic());
   }

   String* mstr = mergestr->asString();
   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(mstr);
   String mr_str( *mstr );
   cstring->unlockInstance(tk1);

   int32 len = ctx->paramCount();
   String *ts = new String;

   // filling the target.
   for( int32 i = 1+first; i <= len ; i ++ )
   {
      Item* item = ctx->param(i-1);

      if ( item->isString() )
      {
         String* src = item->asString();
         InstanceLock::Token* tk1 = cstring->lockInstance(src);
         *ts += *src;
         cstring->unlockInstance(tk1);
      }
      else
      {
         delete ts;
         throw FALCON_SIGN_XERROR( ParamError, e_param_type, .extra("Parameters must be strings") );
      }

      if ( i < len )
      {
         ts->append( mr_str );
      }

      ts->reserve( len/i * ts->size() );
   }

   ctx->returnFrame( FALCON_GC_HANDLE(ts) );
}

static void internal_find( VMContext* ctx, Function* func, bool mode )
{
   // Parameter checking;
   Item *target;
   Item *needle;
   Item *start_item;
   Item *end_item;

   if ( ctx->isMethodic() )
   {
      target = &ctx->self();
      needle = ctx->param(0);
      start_item = ctx->param(1);
      end_item = ctx->param(2);
   }
   else
   {
      target = ctx->param(0);
      needle = ctx->param(1);
      start_item = ctx->param(2);
      end_item = ctx->param(3);
   }

   if ( target == 0 || ! target->isString()
        || needle == 0 || ! needle->isString()
        || (start_item != 0 && ! start_item->isOrdinal())
        || (end_item != 0 && ! end_item->isOrdinal())
        )
   {
      throw func->paramError(__LINE__, SRC, ctx->isMethodic());
   }

   int64 start = start_item == 0 ? 0 : start_item->forceInteger();
   int64 end = end_item == 0 ? -1 : end_item->forceInteger();
   String *sTarget = target->asString();

   int64 len = sTarget->length();
   // negative? -- fix
   if ( start < 0 ) end = len + start + 1;

   // out of range?
   if ( start < 0 || start >= len )
   {
      ctx->returnFrame( -1 );
      return;
   }

   if ( end < 0 ) end = len + end+1;
   // again < than 0? -- it's out of range.
   if ( end < 0 )
   {
      ctx->returnFrame( -1 );
      return;
   }

   if( end > len ) end = len;

   ClassString* cstring = static_cast<ClassString*>(func->methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(target);
   InstanceLock::Token* tk2 = target != needle ? cstring->lockInstance(needle) : 0;

   uint32 pos = mode ?
            target->asString()->rfind( *needle->asString(), (uint32) start, (uint32) end ) :
            target->asString()->find( *needle->asString(), (uint32) start, (uint32) end );

   if ( tk2 !=0 ) cstring->unlockInstance(tk2);
   cstring->unlockInstance(tk1);

   if ( pos != csh::npos )
      ctx->returnFrame( (int64)pos );
   else
      ctx->returnFrame( -1 );

}

/*#
   @method find String
   @brief Finds a substring.
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

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( find, "string:S,needle:S,start:[N],end:[N]" );
FALCON_DEFINE_FUNCTION_P1(find)
{
   internal_find( ctx, this, false );
}

/*#
   @method rfind String
   @brief Finds a substring backwards.
   @param needle Substring to search for.
   @optparam start Optional position from which to start the search in string.
   @optparam end Optional position at which to end the search in string.
   @return The position where the substring is found, or -1.

   Works exactly as @a String.find, except for the fact that the last match
   in the string (or in the specified interval) is returned.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( rfind, "string:S,needle:S,start:[N],end:[N]" );
FALCON_DEFINE_FUNCTION_P1(rfind)

{
   internal_find( ctx, this, true );
}


static void internal_trim( String::t_trimmode mode, Function* func, VMContext* ctx, bool inPlace )
{
   String *self;
   Item *trimChars;

   if ( ctx->isMethodic() )
   {
      self = ctx->self().asString();
      trimChars = ctx->param(0);
   }
   else
   {
      Item *i_str = ctx->param( 0 );
      if ( i_str == 0 || ! i_str->isString() )
      {
         throw func->paramError(__LINE__, SRC, ctx->isMethodic());
      }

      self = i_str->asString();
      trimChars = ctx->param(1);
   }

   if ( trimChars != 0 && trimChars->isArray() && ! trimChars->isString() )
   {
      throw func->paramError(__LINE__, SRC, ctx->isMethodic());
   }

   ClassString* cstring = static_cast<ClassString*>(func->methodOf());

   InstanceLock::Token* tk;
   String *cs;
   if( inPlace )
   {
      if( self->isImmutable() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type, .extra("Immutable string") );
      }
      tk = cstring->lockInstance( self );
      cs = self;
   }
   else {
      tk = cstring->lockInstance( self );
      cs = new String( *self );
      FALCON_GC_HANDLE( cs );
      cstring->unlockInstance( tk );
      tk = 0;
   }

   if ( trimChars == 0 || trimChars->isNil() ) {
      cs->trim( mode );
   }
   else
   {
      cs->trimFromSet( mode, *trimChars->asString() );
   }

   if( tk != 0 )
   {
      cstring->unlockInstance( tk );
   }

   ctx->returnFrame( Item(cs->handler(), cs) );

}


/*#
   @method trim String
   @brief Trims whitespaces from both ends of a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at both ends of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( trim, "string:S,trimSet:[S]" );
FALCON_DEFINE_FUNCTION_P1(trim)
{
   internal_trim( String::e_tm_all, this, ctx, false);
}

/*#
   @method ftrim String
   @brief Trims front whitespaces in a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the beginning of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.

   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( ftrim, "string:S,trimSet:[S]" );
FALCON_DEFINE_FUNCTION_P1( ftrim )
{
   internal_trim(String::e_tm_front, this, ctx, false);
}

/*#
   @method rtrim String
   @brief Trims trailing whitespaces in a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.

   A new string, which is a copy of the original one with all characters in @b trimSet
   at the end of the string removed, is returned. If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns. The
   original string is unmodified.

   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( rtrim, "string:S,trimSet:[S]" );
FALCON_DEFINE_FUNCTION_P1( rtrim )
{
   internal_trim(String::e_tm_back, this, ctx, false);
}



/*#
   @method buffer String
   @brief (static) Pre-allocates a mutable empty string.
   @param size Size of the pre-allocated string.
   @return The new string.

   The returned string is an empty string, and equals to "". However, the required
   size is pre-allocated, and addition to this string (i.e. += operators)
   takes place in a fraction of the time otherwise required, up tho the filling
   of the pre-allocated buffer. Also, this string can be fed into file functions,
   the pre-allocation size being used as the input read size.
*/
FALCON_DECLARE_FUNCTION( buffer, "size:N" );
FALCON_DEFINE_FUNCTION_P1( buffer )
{
   // Parameter checking;
   Item *qty = ctx->param(0);
   if ( qty == 0 || ! qty->isOrdinal() ) {
      throw paramError(__LINE__, SRC );
   }

   int32 size = (int32) qty->forceInteger();
   if ( size <= 0 ) {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ) );
   }

   ctx->returnFrame( FALCON_GC_HANDLE( new String( size ) ) );
}


static void internal_upper_lower( VMContext* ctx, Function* func, bool isUpper, bool inPlace )
{
   Item *source;

   // Parameter checking;
   if ( ctx->isMethodic() )
   {
      source = &ctx->self();
   }
   else
   {
      source = ctx->param(0);
      if ( source == 0 || ! source->isString() ) {
         throw func->paramError(__LINE__, SRC, false);
      }
   }

   ClassString* clstr = static_cast<ClassString*>(func->methodOf());
   String *src = source->asString();

   String* target;

   if( inPlace )
   {
      if( src->isImmutable() )
      {
         throw new ParamError( ErrorParam( e_acc_forbidden, __LINE__, SRC ). extra( "Immutable string ") );
      }

      InstanceLock::Token* tk = clstr->lockInstance(src);
      target = src;
      if( isUpper )
         target->upper();
      else
         target->lower();
      clstr->unlockInstance(tk);
   }
   else
   {
      InstanceLock::Token* tk = clstr->lockInstance(src);
      target = new String(*src);
      clstr->unlockInstance(tk);

      FALCON_GC_HANDLE(target);

      if( isUpper )
         target->upper();
      else
         target->lower();
   }

   ctx->returnFrame( Item(target->handler(), target) );
}

/*#
   @method upper String
   @brief Returns an upper case version of this string.
   @return The uppercased string.

   All the Latin characters in the string are turned uppercase. Other characters
   are left untouched.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( upper, "string:S" );
FALCON_DEFINE_FUNCTION_P1( upper )
{
   internal_upper_lower(ctx, this, true, false );
}

/*#
   @method lower String
   @brief Returns a lowercase version of this string.
   @return The lowercased string.

   All the Latin characters in the string are turned lowercase. Other characters
   are left untouched.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( lower, "string:S" );
FALCON_DEFINE_FUNCTION_P1( lower )
{
   internal_upper_lower(ctx, this, false, false );
}


static void internal_start_end_with( VMContext* ctx, Function* func, bool isEnd  )
{
   Item* source;
   Item* i_token;
   Item* i_icase;

   // Parameter checking;
   if ( ctx->isMethodic() )
   {
      source = &ctx->self();
      i_token = ctx->param(0);
      i_icase = ctx->param(1);
   }
   else
   {
      source = ctx->param(0);
      i_token = ctx->param(1);
      i_icase = ctx->param(2);
   }

   if ( source == 0 || ! source->isString() ||
        i_token == 0 || ! i_token->isString() )
   {
      throw func->paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   String *src = source->asString();
   bool iCase = i_icase ? i_icase->isTrue() : false;

   ClassString* cstring = static_cast<ClassString*>(func->methodOf());

   InstanceLock::Token* tk = cstring->lockInstance(src);

   bool res = isEnd ?
            src->endsWith( *i_token->asString(), iCase ) :
            src->startsWith( *i_token->asString(), iCase );

   cstring->unlockInstance(tk);

   ctx->returnFrame( Item().setBoolean(res) );
}

/*#
   @method startsWith String
   @brief Check if a strings starts with a substring.
   @param token The substring that will be compared with this string.
   @optparam icase If true, performs a case neutral check
   @return True if @b token matches the beginning of this string, false otherwise.

   This method performs a comparison check at the beginning of the string.
   If this string starts with @b token, the function returns true. If @b token
   is larger than the string, the method will always return false, and
   if @b token is an empty string, it will always match.

   The optional parameter @b icase can be provided as true to have this
   method to perform a case insensitive match.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( startsWith, "string:S,token:S,icase:[B]" );
FALCON_DEFINE_FUNCTION_P1( startsWith )
{
   internal_start_end_with( ctx, this, false );
}


/*#
   @method endsWith String
   @brief Check if a strings ends with a substring.
   @param token The substring that will be compared with this string.
   @optparam icase If true, performs a case neutral check
   @return True if @b token matches the end of this string, false otherwise.

   This method performs a comparison check at the end of the string.
   If this string ends with @b token, the function returns true. If @b token
   is larger than the string, the method will always return false, and
   if @b token is an empty string, it will always match.

   The optional parameter @b icase can be provided as true to have this
   method to perform a case insensitive match.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( endsWith, "string:S,token:S,icase:[B]" );
FALCON_DEFINE_FUNCTION_P1( endsWith )
{
   internal_start_end_with( ctx, this, true );
}


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

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( cmpi, "this:S,string:S" );
FALCON_DEFINE_FUNCTION_P1( cmpi )
{
   Item *s1_itm, *s2_itm;

   // Parameter checking;
   if( ctx->isMethodic() )
   {
      s1_itm = &ctx->self();
      s2_itm = ctx->param(0);
   }
   else
   {
      s1_itm = ctx->param(0);
      s2_itm = ctx->param(1);
   }

   if ( s1_itm == 0 || ! s1_itm->isString() || s2_itm == 0 || !s2_itm->isString() )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   String* str1 = s1_itm->asString();
   String* str2 = s2_itm->asString();
   // avoid double lock of the same item.
   if( str1 == str2 )
   {
      ctx->returnFrame((int64)0);
      return;
   }

   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(str1);
   InstanceLock::Token* tk2 = str1 != str2 ? cstring->lockInstance(str2) : 0;

   int32 result = str1->compareIgnoreCase(*str2);

   if( tk2 != 0 ) cstring->unlockInstance(tk2);
   cstring->unlockInstance(tk1);

   ctx->returnFrame((int64) result );
}



inline void internal_escape( VMContext* ctx, Function* func, int mode )
{
   Item *i_string;

   // Parameter checking;
   if( ctx->isMethodic() )
   {
      i_string = &ctx->self();
   }
   else
   {
      i_string = ctx->param(0);
      if ( i_string == 0 || ! i_string->isString() )
      {
         throw func->paramError(__LINE__, SRC, false );
      }
   }

   String* source = i_string->asString();
   ClassString* cstring = static_cast<ClassString*>(func->methodOf());
   InstanceLock::Token* tk;

   String* str;

   if( mode >= 2 )
   {
      str = new String;
      FALCON_GC_HANDLE(str);
      tk = cstring->lockInstance(source);
   }
   else {
      tk = cstring->lockInstance(source);
      str = new String(*source);
      cstring->unlockInstance(tk);
      tk = 0;
      FALCON_GC_HANDLE(str);
   }

   switch( mode )
   {
   case 0: //esq
      str->escapeQuotes();
      break;

   case 1: //unesq
      str->unescapeQuotes();
      break;

   case 2: //escape-full
      source->escape( *str );
      break;

   case 3: // escape
      source->escapeFull( *str );
      break;

   case 4: //unescape
      source->unescape( *str );
      break;
   }

   if( tk != 0 ) cstring->unlockInstance(tk);

   ctx->returnFrame( Item(str->handler(), str) );
}

/*#
   @method esq String
   @brief Escapes the double quotes in this string.
   @return A new string with the quotes escaped.

   This method returns a new string where all the quotes ('),
   double quotes (") and backslashes (\\) are preceded by a single
   backslash (\\). This makes the string parsable by the vast
   majority of string parsing languages, including Falcon itself.

   @note When used statically, this method takes a target string as first parameter.

   @see String.unesq
*/

FALCON_DECLARE_FUNCTION( esq, "string:S" );
FALCON_DEFINE_FUNCTION_P1( esq )
{
   internal_escape( ctx, this, 0 );
}

/*#
   @method unesq String
   @brief Unescapes backslash-based escape quote sequences.
   @return A new string with backslash sequences unescaped.

   This method unescapes backslash-quote sequences only;
   backslash sequences having a special meaning in many
   contexts as i.e. '\\r' are not expanded.

   The @a String.unescape method escapes also other sequences
   as the Falcon parser would.

   @note When used statically, this method takes a target string as first parameter.

   @see String.esq

*/

FALCON_DECLARE_FUNCTION( unesq, "string:S" );
FALCON_DEFINE_FUNCTION_P1( unesq )
{
   internal_escape( ctx, this, 1 );
}


/*#
   @method escape String
   @brief Escapes all the special characters in the string up to UNICODE 127.
   @return A new string with escaped contents.

   This method escapes special characters in the string
   up to UNICODE 127.
   Characters as the new-line are translated into common
   sequences that are understood by the falcon interpreter as '\\n'.

   The @a String.escapeFull method escapes also characters above
   UNICODE 127, encoding them in a \\xNNNN hexadecimal unicode
   value sequence.

   @note When used statically, this method takes a target string as first parameter.

   @see String.esq
   @see strEsq
   @see strUnescape
*/

FALCON_DECLARE_FUNCTION( escape, "string:S" );
FALCON_DEFINE_FUNCTION_P1( escape )
{
   internal_escape( ctx, this, 2 );
}


/*#
   @method escapeFull String
   @brief Escapes all the special characters in the string.
   @return A new string with escaped contents.

   This method escapes special characters in the string.
   Characters as the new-line are translated into common
   sequences that are understood by the falcon interpreter as '\\n'.

   This method escapes also characters above
   UNICODE 127, encoding them in a \\xNNNN hexadecimal UNICODE
   value sequence.

   @note When used statically, this method takes a target string as first parameter.

   @see String.esq
   @see String.escape
*/

FALCON_DECLARE_FUNCTION( escapeFull, "string:S" );
FALCON_DEFINE_FUNCTION_P1( escapeFull )
{
   internal_escape( ctx, this, 3 );
}


/*#
   @method unescape String
   @brief Unescapes all the special characters in the string.
   @return A new strings with the special backslash sequences unescaped.

   @note When used statically, this method takes a target string as first parameter.

   @see String.esq
   @see String.escape
*/

FALCON_DECLARE_FUNCTION( unescape, "string:S" );
FALCON_DEFINE_FUNCTION_P1( unescape )
{
   internal_escape( ctx, this, 4 );
}


/*#
   @method replace String
   @brief Replaces the all the occurrences of a substring with another one.
   @param substr The substring that will be replaced.
   @param repstr The string that will take the place of substr.
   @optparam count Maximum number of substitutions.
   @return A copy of the string with the occurrences of the searched substring replaced.

   @note When used statically, this method takes a target string as first parameter.
 */
FALCON_DECLARE_FUNCTION( replace, "string:S,substr:S,repstr:S,count:N" );
FALCON_DEFINE_FUNCTION_P1( replace )
{
   // Parameter checking;
   Item *i_target, *i_needle, *i_replacer, *i_count;

   if( ctx->isMethodic() )
   {
      i_target = &ctx->self();
      i_needle = ctx->param(0);
      i_replacer = ctx->param(1);
      i_count = ctx->param(2);
   }
   else
   {
      i_target = ctx->param(0);
      i_needle = ctx->param(1);
      i_replacer = ctx->param(2);
      i_count = ctx->param(3);
   }

   if ( i_target == 0 || ! i_target->isString()
        || i_needle == 0 || ! i_needle->isString()
        || i_replacer == 0 || ! i_replacer->isString()
        || (i_count != 0 && ! i_count->isOrdinal())
        )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   // Parameter estraction.
   String *tg_str = i_target->asString();
   String *ned_str = i_needle->asString();
   String *rep_str = i_replacer->asString();
   int32 count = i_count == 0 ? 0 : (int32) i_count->forceInteger();

   String* str = new String;

   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(tg_str);
   InstanceLock::Token* tk2 = tg_str != ned_str ? cstring->lockInstance(ned_str) : 0;
   InstanceLock::Token* tk3 = rep_str != tg_str && rep_str != ned_str ? cstring->lockInstance(rep_str) : 0;

   tg_str->replace( *ned_str, *rep_str, *str, count );

   if( tk3 != 0 ) cstring->unlockInstance(tk3);
   if( tk2 != 0 ) cstring->unlockInstance(tk2);
   cstring->unlockInstance(tk1);

   ctx->returnFrame( FALCON_GC_HANDLE(str) );
}

/*#
   @method substr String
   @brief Replaces the all the occurrences of a substring with another one.
   @param start The substring that will be replaced.
   @optparam length The substring that will be replaced.
   @return A copy of the string with the occurrences of the searched substring replaced.

   @note When used statically, this method takes a target string as first parameter.
 */
FALCON_DECLARE_FUNCTION( substr, "string:S,start:N,length:[N]" );
FALCON_DEFINE_FUNCTION_P1( substr )
{
   // Parameter checking;
   Item *i_string, *i_start, *i_length;

   if( ctx->isMethodic() )
   {
      i_string = &ctx->self();
      i_start = ctx->param(0);
      i_length = ctx->param(1);
   }
   else
   {
      i_string = ctx->param(0);
      i_start = ctx->param(1);
      i_length = ctx->param(2);
   }

   if ( i_string == 0 || ! i_string->isString()
          || i_start == 0 || ! i_start->isOrdinal()
          || (i_length != 0 && ! i_length->isOrdinal())
          )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   String* str = i_string->asString();
   int64 start = i_start == 0 ? 0 : i_start->forceInteger();
   if( start < 0 )
   {
      start = 0;
   }

   int64 length = 0;
   if( i_length == 0 )
   {
      length = str->length()-start;
   }
   else
   {
      length = i_length->forceInteger();
      if( length < 0 )
      {
         length = str->length();
      }
   }

   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(str);
   String* ret = new String(str->subString(static_cast<int32>(start), static_cast<int32>(start + length) ));
   cstring->unlockInstance(tk1);

   ctx->returnFrame(FALCON_GC_HANDLE(ret));
}


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

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( wmatch, "string:S,wildcard:S,ignoreCase:[B]" );
FALCON_DEFINE_FUNCTION_P1( wmatch )
{
   // Parameter checking;
   Item *s1_itm, *s2_itm, *i_bIcase;

   if ( ctx->isMethodic() )
   {
      s1_itm = &ctx->self();
      s2_itm = ctx->param(0);
      i_bIcase = ctx->param(1);
   }
   else
   {
      s1_itm = ctx->param(0);
      s2_itm = ctx->param(1);
      i_bIcase = ctx->param(2);
   }

   if ( s1_itm == 0 || ! s1_itm->isString() || s2_itm == 0 || !s2_itm->isString() ) {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   // Ignore case?
   bool bIcase = i_bIcase == 0 ? false : i_bIcase->isTrue();

   // The first is the wildcard, the second is the matched thing.
   String *cfr = s1_itm->asString();
   String *wcard = s2_itm->asString();

   ClassString* cstring = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstring->lockInstance(cfr);
   InstanceLock::Token* tk2 = cfr != wcard ? cstring->lockInstance(wcard) : 0;

   bool bResult = cfr->wildcardMatch( *wcard, bIcase );

   if( tk2 != 0 ) cstring->unlockInstance(tk2);
   cstring->unlockInstance(tk1);

   ctx->returnFrame( Item().setBoolean( bResult ) );
}


template< class __Checker>
void internal_checkType( VMContext* ctx, Function* func, const __Checker& checker )
{
   // Parameter checking;
   Item *i_string;

   if( ctx->isMethodic() )
   {
      i_string = &ctx->self();
   }
   else
   {
      i_string = ctx->param(0);
      if ( i_string == 0 )
      {
         throw func->paramError(__LINE__, SRC );
      }
   }

   bool retval;

   if( i_string->isString() )
   {
      ClassString* scls = static_cast<ClassString*>( func->methodOf() );
      String* string = i_string->asString();

      InstanceLock::Token* tk = scls->lockInstance( string );
      retval = checker(string);
      scls->unlockInstance(tk);
   }
   else if( i_string->isOrdinal() )
   {
      retval = checker( i_string->forceInteger() );
   }
   else
   {
      // can't be methodic if we're here.
      throw func->paramError(__LINE__, SRC );
   }

   ctx->returnFrame( Item().setBoolean( retval ) );
}

class AlphaChecker {
public:
   bool operator()( String* str ) const { return str->isAlpha(); }
   bool operator()( int64 num ) const { return String::isAlpha( static_cast<char_t>(num) ); }
};

class DigitChecker {
public:
   bool operator()( String* str ) const { return str->isDigit(); }
   bool operator()( int64 num ) const { return String::isDigit( static_cast<char_t>(num) ); }
};

class isAlphaNumChecker {
public:
   bool operator()( String* str ) const { return str->isAlphaNum(); }
   bool operator()( int64 num ) const { return String::isAlphaNum( static_cast<char_t>(num) ); }
};

class PunctChecker {
public:
   bool operator()( String* str ) const { return str->isPunct(); }
   bool operator()( int64 num ) const { return String::isPunct( static_cast<char_t>(num) ); }
};

class UpperChecker {
public:
   bool operator()( String* str ) const { return str->isUpper(); }
   bool operator()( int64 num ) const { return String::isUpper( static_cast<char_t>(num) ); }
};

class isLowerChecker {
public:
   bool operator()( String* str ) const { return str->isLower(); }
   bool operator()( int64 num ) const { return String::isLower( static_cast<char_t>(num) ); }
};

class isWhitespaceChecker {
public:
   bool operator()( String* str ) const { return str->isWhitespace(); }
   bool operator()( int64 num ) const { return String::isWhitespace( static_cast<char_t>(num) ); }
};

class isPrintableChecker {
public:
   bool operator()( String* str ) const { return str->isPrintable(); }
   bool operator()( int64 num ) const { return String::isPrintable( static_cast<char_t>(num) ); }
};

class isASCIIChecker {
public:
   bool operator()( String* str ) const { return str->isASCII(); }
   bool operator()( int64 num ) const { return String::isASCII( static_cast<char_t>(num) ); }
};

class isISOChecker {
public:
   bool operator()( String* str ) const { return str->isISO(); }
   bool operator()( int64 num ) const { return String::isISO( static_cast<char_t>(num) ); }
};

/*#
   @method isAlpha String
   @brief Checks if the string is composed of Latin alphabet characters only
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isAlpha, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isAlpha )
{
   internal_checkType<AlphaChecker>( ctx, this, AlphaChecker() );
}

/*#
   @method isDigit String
   @brief Checks if the string is composed of Arabic ciphers only
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isDigit, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isDigit )
{
   internal_checkType( ctx, this, DigitChecker() );
}

/*#
   @method isAlphaNum String
   @brief Checks if the string is composed of Latin letters or Arabic ciphers only.
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isAlphaNum, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isAlphaNum )
{
   internal_checkType( ctx, this, isAlphaNumChecker() );
}

/*#
   @method isPunct String
   @brief Checks if the string is composed of puntaction characters.
   @return True if the check succeeds.

   Characters known are '.', ',', ':', ';', '?', '!'.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isPunct, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isPunct )
{
   internal_checkType( ctx, this, PunctChecker() );
}

/*#
   @method isUpper String
   @brief Checks if the string is composed of upper case Latin letters only.
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isUpper, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isUpper )
{
   internal_checkType( ctx, this, UpperChecker() );
}


/*#
   @method isLower String
   @brief Checks if the string is composed of lower case Latin letters only.
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isLower, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isLower )
{
   internal_checkType( ctx, this, isLowerChecker() );
}

/*#
   @method isWhitespace String
   @brief Checks if the string is composed of space, tabulation, newline and return characters.
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isWhitespace, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isWhitespace )
{
   internal_checkType( ctx, this, isWhitespaceChecker() );
}

/*#
   @method isPrintable String
   @brief Checks if the string is composed of characters in the printable range.
   @return True if the check succeeds.

   This includes all the characters between the blank character (UNICODE 32) and
   the ASCII 127 character (UNICODE 127), plus tabulation and new line.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isPrintable, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isPrintable )
{
   internal_checkType( ctx, this, isPrintableChecker() );
}

/*#
   @method isASCII String
   @brief Checks if the string is composed of characters between UNICODE 0 and 127 included.
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isASCII, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isASCII )
{
   internal_checkType( ctx, this, isASCIIChecker() );
}

/*#
   @method isISO String
   @brief Checks if the string is composed of characters between UNICODE 0 and 255 included.
   @return True if the check succeeds.

   @note When used statically, this method takes a target string as first parameter, or
   a UNICODE number representing a single character.
 */
FALCON_DECLARE_FUNCTION( isISO, "string:S|N" );
FALCON_DEFINE_FUNCTION_P1( isISO )
{
   internal_checkType( ctx, this, isISOChecker() );
}


//=======================================================================================
// Mutable Strings
//


/*#
   @method atrim String
   @brief Trims whitespaces from both ends of a string in place.
   @optparam trimSet A set of characters that must be removed.
   @return The same string.

   This method removes all characters in @b trimSet at the beginning and
   at the end of the string.
   If @b trimSet is not supplied, it defaults to space, tabulation characters,
   new lines and carriage returns.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( atrim, "trimSet:[S]" );
FALCON_DEFINE_FUNCTION_P1(atrim)
{
   internal_trim( String::e_tm_all, this, ctx, true);
}

/*#
   @method aftrim String
   @brief Trims front whitespaces in a string in place.
   @optparam trimSet A set of characters that must be removed.
   @return This string.

   This method removes all characters in @b trimSet at the beginning of the string.
   If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns.

   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( aftrim, "trimSet:[S]" );
FALCON_DEFINE_FUNCTION_P1( aftrim )
{
   internal_trim(String::e_tm_front, this, ctx, true);
}

/*#
   @method rtrim String
   @brief Trims trailing whitespaces in a string.
   @optparam trimSet A set of characters that must be removed.
   @return The trimmed version of the string.

   This method removes all characters in @b trimSet at the end of the string.
   If @b trimSet is not supplied, it
   defaults to space, tabulation characters, new lines and carriage returns.

   @note When used statically, this method takes a target string as first parameter.
*/

FALCON_DECLARE_FUNCTION( artrim, "trimSet:[S]" );
FALCON_DEFINE_FUNCTION_P1( artrim )
{
   internal_trim(String::e_tm_back, this, ctx, true);
}


/*#
   @method aupper String
   @brief Transforms this mutable string in uppercase.
   @return This same string

   All the Latin characters in the string are turned uppercase. Other characters
   are left untouched.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( aupper, "string:S" );
FALCON_DEFINE_FUNCTION_P1( aupper )
{
   internal_upper_lower(ctx, this, true, true );
}


/*#
   @method alower String
   @brief Transforms this mutable string in lowercase.
   @return This same string

   All the Latin characters in the string are turned lowercase. Other characters
   are left untouched.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( alower, "string:S" );
FALCON_DEFINE_FUNCTION_P1( alower )
{
   internal_upper_lower(ctx, this, false, true );
}


/*#
   @method fill String
   @brief Fills a string with a given character or substring.
   @Prime chr The character (unicode value) or substring used to refill this string.
   @return The string itself.

   This method fills the physical storage of the given string with a single
   character or a repeated substring. This can be useful to clean a string used repeatedly
   as input buffer.

   @note When used statically as a class method, the first parameter can be a mutable string.
*/

FALCON_DECLARE_FUNCTION( fill, "target:MString,chr:N|S" );
FALCON_DEFINE_FUNCTION_P1( fill )
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

   if( i_string == 0 || ! i_string->isString()
      || i_chr == 0 || ( ! i_chr->isOrdinal() && !i_chr->isString())
      )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic() );
   }

   String *string = i_string->asString();

   if( string->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_acc_forbidden, __LINE__, SRC ).extra("Immutable string") );
   }

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

/*#
   @method insert String
   @brief Inserts a string into an an existing string in place.
   @param pos Position where to insert the string
   @param needle String to be inserted
   @return This same string.

   This method inserts a @b needle string at a given position in
   the target string (before the character currently being in that position).
   The original string is modified in place; this
   means that the original string must be mutable.

   If @b pos is negative, the position is considered relative to the end of the
   string, and the insertion will be done @b after the given position. So,
   if pos is -1, the string will be actually appended to the original one, if it's
   -2, it will be inserted before the last character, and if it's -(string.len+1), it
   will be placed in front of the first character.

   If @b pos is out of range, a ParamError will be raised.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( insert, "pos:N,needle:S" );
FALCON_DEFINE_FUNCTION_P1(insert)
{
   Item *i_string, *i_pos, *i_needle;

   // Parameter checking;
   ctx->getMethodicParams( i_string, i_pos, i_needle );

   if( i_string == 0 || ! i_string->isString()
      || i_pos == 0 || ! i_pos->isOrdinal()
      || i_needle == 0 || ! i_needle->isString()
      )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic() );
   }

   String *string = i_string->asString();

   if( string->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_acc_forbidden, __LINE__, SRC ).extra("Immutable string") );
   }

   String* needle = i_needle->asString();
   int64 pos = i_pos->forceInteger();

   ClassString* cstr = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstr->lockInstance(string);
   if( pos < 0 )
   {
      pos = string->length() + pos +1;
   }
   if( pos < 0 || pos > string->length() )
   {
      cstr->unlockInstance(tk1);
      throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("Invalid insert position"));
   }

   InstanceLock::Token* tk2 = needle != string ? cstr->lockInstance(needle) : 0;
   string->insert(static_cast<length_t>(pos),0,*needle);
   if( tk2 != 0 ) cstr->unlockInstance(tk2);
   cstr->unlockInstance(tk1);

   ctx->returnFrame( Item(string->handler(), string) );
}


/*#
   @method remove String
   @brief Remove some characters from the string in place.
   @param pos Position where to remove the characters
   @param count Number of characters to be removed.
   @return This same string.

   This method removes a given count of characters string at a given position in
   the target string.
   The original string is modified in place; this
   means that the original string must be mutable.

   If @b pos is negative, the position is considered relative to the end of the
   string. So, if pos is -1, the last character will be removed.

   If @b pos is out of range, a ParamError will be raised.

   If count is negative, everything will be removed up to the end of the string;
   if it's 0, nothing will be removed.

   @note When used statically, this method takes a target string as first parameter.
*/
FALCON_DECLARE_FUNCTION( remove, "pos:N,count:S" );
FALCON_DEFINE_FUNCTION_P1(remove)
{
   Item *i_string, *i_pos, *i_count;

   // Parameter checking;
   ctx->getMethodicParams( i_string, i_pos, i_count );

   if( i_string == 0 || ! i_string->isString()
      || i_pos == 0 || ! i_pos->isOrdinal()
      || i_count == 0 || ! i_count->isOrdinal()
      )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic() );
   }

   String *string = i_string->asString();

   if( string->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_acc_forbidden, __LINE__, SRC ).extra("Immutable string") );
   }

   int64 pos = i_pos->forceInteger();
   int64 count = i_count->forceInteger();

   ClassString* cstr = static_cast<ClassString*>(methodOf());
   InstanceLock::Token* tk1 = cstr->lockInstance(string);

   if( pos < 0 )
   {
      pos = string->length() + pos +1;
   }
   if( pos < 0 || pos > string->length() )
   {
      cstr->unlockInstance(tk1);
      throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("Invalid insert position"));
   }
   if( count < 0 )
   {
      count = string->length();
   }

   string->remove(static_cast<length_t>(pos), static_cast<length_t>(count));
   cstr->unlockInstance(tk1);

   ctx->returnFrame( Item(string->handler(), string) );
}

}

//==============================================================================
// Class properties used for enumeration
//

ClassString::ClassString():
   Class( "String", FLC_CLASS_ID_STRING )
{
   init();
}


ClassString::ClassString( const String& subname ):
         Class( subname, FLC_CLASS_ID_STRING )
{
   init();
}


void ClassString::init()
{
   m_initNext = new PStepInitNext;
   m_nextOp = new PStepNextOp(this);

   addProperty( "isText", &get_isText, &set_isText );
   addProperty( "len", &get_len );
   addProperty( "mutable", &get_mutable );
   addProperty( "allocated", &get_allocated );
   addProperty( "charSize", &get_charSize, &set_charSize );

   addProperty( "allAlpha", &get_allAlpha, 0, true, true );
   addProperty( "allAlphaNum", &get_allAlphaNum, 0, true, true );
   addProperty( "allDigit", &get_allDigit, 0, true, true );
   addProperty( "allUpper", &get_allUpper, 0, true, true );
   addProperty( "allLower", &get_allLower, 0, true, true );
   addProperty( "allPunct", &get_allPunct, 0, true, true );

   addMethod( new _classString::Function_front, true );
   addMethod( new _classString::Function_back, true );
   addMethod( new _classString::Function_substr, true );
   addMethod( new _classString::Function_split, true );
   addMethod( new _classString::Function_splittr, true );
   addMethod( new _classString::Function_merge, true );
   addMethod( new _classString::Function_join, true );

   addMethod( new _classString::Function_find, true );
   addMethod( new _classString::Function_rfind, true );
   addMethod( new _classString::Function_wmatch, true );

   addMethod( new _classString::Function_trim, true );
   addMethod( new _classString::Function_ftrim, true );
   addMethod( new _classString::Function_rtrim, true );

   addMethod( new _classString::Function_upper, true );
   addMethod( new _classString::Function_lower, true );

   addMethod( new _classString::Function_startsWith, true );
   addMethod( new _classString::Function_endsWith, true );
   addMethod( new _classString::Function_cmpi, true );

   addMethod( new _classString::Function_escape, true );
   addMethod( new _classString::Function_escapeFull, true );
   addMethod( new _classString::Function_unescape, true );
   addMethod( new _classString::Function_esq, true );
   addMethod( new _classString::Function_unesq, true );

   addMethod( new _classString::Function_replace, true );

   addMethod( new _classString::Function_isAlpha, true );
   addMethod( new _classString::Function_isDigit, true );
   addMethod( new _classString::Function_isAlphaNum, true );
   addMethod( new _classString::Function_isPunct, true );
   addMethod( new _classString::Function_isUpper, true );
   addMethod( new _classString::Function_isLower, true );
   addMethod( new _classString::Function_isWhitespace, true );
   addMethod( new _classString::Function_isPrintable, true );
   addMethod( new _classString::Function_isASCII, true );
   addMethod( new _classString::Function_isISO, true );
   //====================================================

   addMethod( new _classString::Function_atrim, true );
   addMethod( new _classString::Function_aftrim, true );
   addMethod( new _classString::Function_artrim, true );

   addMethod( new _classString::Function_aupper, true );
   addMethod( new _classString::Function_alower, true );
   addMethod( new _classString::Function_fill, true );

   addMethod( new _classString::Function_insert, true );
   addMethod( new _classString::Function_remove, true );

   addMethod( new _classString::Function_buffer, true );

   m_modelAllAlpha = new String( "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
   m_modelAllAlpha->setImmutable(true);

   m_modelAllAlphaNum = new String( "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
   m_modelAllAlphaNum->setImmutable(true);

   m_modelAllDigit = new String( "0123456789");
   m_modelAllDigit->setImmutable(true);

   m_modelAllUpper = new String( "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
   m_modelAllUpper->setImmutable(true);

   m_modelAllLower = new String( "abcdefghijklmnopqrstuvwxyz");
   m_modelAllLower->setImmutable(true);

   m_modelAllPunct = new String( ".,;:!?");
   m_modelAllPunct->setImmutable(true);
}


ClassString::~ClassString()
{
   delete m_initNext;
   delete m_nextOp;
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
   String* orig = static_cast<String*>( data );
   InstanceLock::Token* tk = m_lock.lock(data);
   String value(*orig);
   value.setImmutable(orig->isImmutable());
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

      if ( ! static_cast<String*>( instance )->isImmutable() ){
         target.append("m");
      }

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

      if(maxlen >0 && pos*3 >= (unsigned int) maxlen)
      {
         target.append( " ... " );
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

      ctx->stackResult( 2, FALCON_GC_HANDLE(copy) );

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

      ctx->stackResult( 2, FALCON_GC_HANDLE(copy) );

      return;
   }

   // else we surrender, and we let the virtual system to find a way.
   ctx->pushCode( m_nextOp );

   // this will transform op2 slot into its string representation.
   cls->op_toString( ctx, inst );

   if ( ! ctx->wentDeep( m_nextOp ) )
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
         ctx->pushCode( m_initNext );
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
   result->reserve(static_cast<length_t>(copy.size() * count));
   for( int64 i = 0; i < count; ++i )
   {
      result->append(copy);
   }

   ctx->topData() = FALCON_GC_HANDLE(result);
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



   String* self = static_cast<String*>(instance);
   InstanceLock::Token* tk = m_lock.lock(self);
   uint32 chr = self->empty() ? 0 : self->getCharAt(self->length()-1);
   m_lock.unlock(tk);

   count = chr + count;
   if ( count < 0 || count >= 0xFFFFFFFFLL )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "Operator / out of range" ) );
   }
   String* target = new String(1);
   target->append((uint32) count);
   ctx->topData() = FALCON_GC_HANDLE( target );
}


void ClassString::op_mod( VMContext* ctx, void* instance ) const
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
         ctx->stackResult( 2, FALCON_GC_HANDLE(s) );
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
      index->forceClassInst( cls, udata );

      // Confirm we have a range
      if ( cls->typeID() != FLC_CLASS_ID_RANGE )
      {
         throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "Range" ) );
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

      ctx->stackResult( 2, FALCON_GC_HANDLE(s) );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "invalid index" ) );
   }
}


//=======================================================================
// Comparison
//

void ClassString::op_compare( VMContext* ctx, void* self ) const
{
   Item* op1, *op2;

   OpToken token( ctx, op2, op1 );

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


void ClassString::op_in( VMContext* ctx, void* instance ) const
{
   if( ! ctx->topData().isString() )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "S" ) );
      return;
   }


   /* No lock -- we can accept sub-program level uncertainty */
   String* self = static_cast<String*>(instance);
   String* other = ctx->topData().asString();
   ctx->popData();
   if( other == self )
   {
      ctx->topData().setBoolean(true);
      return;
   }

   InstanceLock::Token* l1 = m_lock.lock(self);
   InstanceLock::Token* l2 = m_lock.lock(other);

   length_t pos = self->find(*other);

   m_lock.unlock(l2);
   m_lock.unlock(l1);

   ctx->topData().setBoolean( pos != String::npos );
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

//========================================================================================
// Mutable operators
//

void ClassString::op_aadd( VMContext* ctx, void* self ) const
{
   String* str = static_cast<String*>( self );
   if( str->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("Immutable string") );
   }

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

void ClassString::op_amul( VMContext* ctx, void* instance ) const
{
   String* self = static_cast<String*>(instance);
   if( self->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("Immutable string") );
   }

   // self count => self
   Item& i_count = ctx->topData();
   if( ! i_count.isOrdinal() )
   {
      throw new OperandError( ErrorParam( e_op_params, __LINE__ ).extra( "N" ) );
   }

   int64 count = i_count.forceInteger();
   ctx->popData();

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
      target->reserve( static_cast<length_t>(target->size() * count));
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


void ClassString::op_amod( VMContext* ctx, void* instance ) const
{
   String* self = static_cast<String*>(instance);
   if( self->isImmutable() )
   {
      throw new OperandError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("Immutable string") );
   }

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

   InstanceLock::Token* tk = m_lock.lock(self);
   self->append((char_t) count);
   m_lock.unlock(tk);
}


void ClassString::op_setIndex( VMContext* ctx, void* self ) const
{
   String& str = *static_cast<String*>( self );
   if( str.isImmutable() )
   {
      throw new OperandError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("Immutable string") );
   }

   Item* value, *arritem, *index;
   ctx->operands( value, arritem, index );

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
