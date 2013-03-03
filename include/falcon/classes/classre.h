/*
   FALCON - The Falcon Programming Language.
   FILE: classre.h

   RE2 object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Feb 2013 13:49:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSRE_H_
#define _FALCON_CLASSRE_H_

#include <falcon/setup.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/string.h>

#include <falcon/pstep.h>
namespace Falcon
{

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
 */

class FALCON_DYN_CLASS ClassRE: public ClassUser
{
public:

   ClassRE();
   virtual ~ClassRE();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext*, DataWriter* dw, void* data ) const;
   virtual void restore( VMContext* , DataReader* dr ) const;

   virtual void describe( void* instance, String& target, int, int ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=============================================================
   virtual bool op_init( VMContext* ctx, void*, int32 pcount ) const;

   virtual void op_div( VMContext* ctx, void* instance ) const;
   virtual void op_mod( VMContext* ctx, void* instance ) const;
   virtual void op_mul( VMContext* ctx, void* instance ) const;
   virtual void op_pow( VMContext* ctx, void* instance ) const;

private:

   //====================================================
   // Properties.
   //

   FALCON_DECLARE_PROPERTY( captures );
   FALCON_DECLARE_PROPERTY( caseSensitive );
   FALCON_DECLARE_PROPERTY( groupNames );

   /*#
    @method match RE
    @param target The string to be matched.
    @return True if the string matches somewhere with the
    regular expression.
   */
   FALCON_DECLARE_METHOD( match, "target:S" );

   /*#
    @method grab RE
    @param target The string to be matched.
    @return If the string matches, returns the matched content,
       otherwise returns nil
   */
   FALCON_DECLARE_METHOD( grab, "target:S" );


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
   FALCON_DECLARE_METHOD( find, "target:S,begin_:[N],end_[N]" );

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
   FALCON_DECLARE_METHOD( findAll, "target:S,begin_:[N],end_[N]" );


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
   FALCON_DECLARE_METHOD( range, "target:S,begin_:[N],end_[N]" );

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
   FALCON_DECLARE_METHOD( rangeAll, "target:S,begin_:[N],end_[N]" );

   /*#
    @method capture RE
    @brief Returns the captured expressions.
    @param target The string to be matched.
    @optparam getAll If true, the first returned element is the whole match,
       otherwise, only parenthezized expressions are returned.
    @return If the string matches, returns an array with the captured expressions,
   */
   FALCON_DECLARE_METHOD( capture, "target:S,getAll:[B]" );

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

   FALCON_DECLARE_METHOD( replace, "target:S,replacer:S" );

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

   FALCON_DECLARE_METHOD( replaceFirst, "target:S,replacer:S" );

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

   FALCON_DECLARE_METHOD( substitute, "target:S,replacer:S" );

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
   FALCON_DECLARE_METHOD( change, "target:S,replacer:S" );

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
   FALCON_DECLARE_METHOD( changeFirst, "target:S,replacer:S" );

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

   FALCON_DECLARE_METHOD( chop, "target:S,replacer:S" );

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
   FALCON_DECLARE_METHOD( consume, "target:S,getAll:[B]" );

   /*#
    @method consume RE
    @brief Cut the string up to where the pattern is matched,
       and return the matched pattern.
    @param target The string that is to be consumed.
    @return on success, the matched pattern. On failure, nil.
   */
   FALCON_DECLARE_METHOD( consumeMatch, "target:S" );

   /*
   FALCON_DECLARE_PROPERTY( back );
   FALCON_DECLARE_PROPERTY( charSize );
   FALCON_DECLARE_PROPERTY( escape );
   FALCON_DECLARE_PROPERTY( esq );
   FALCON_DECLARE_PROPERTY( ftrim );
   FALCON_DECLARE_PROPERTY( isText );
   FALCON_DECLARE_PROPERTY( len );
   FALCON_DECLARE_PROPERTY( lower );
   FALCON_DECLARE_PROPERTY( rtrim );
   FALCON_DECLARE_PROPERTY( trim );
   FALCON_DECLARE_PROPERTY( unescape );
   FALCON_DECLARE_PROPERTY( unesq );
   FALCON_DECLARE_PROPERTY( upper );

   FALCON_DECLARE_METHOD( cmpi, "S" );
   FALCON_DECLARE_METHOD( endsWith, "S" );
   FALCON_DECLARE_METHOD( fill, "S" );
   FALCON_DECLARE_METHOD( join, "..." );
   FALCON_DECLARE_METHOD( merge, "A" );
   FALCON_DECLARE_METHOD( replace, "S,S" );
   FALCON_DECLARE_METHOD( replicate, "N" );
   FALCON_DECLARE_METHOD( rfind, "S" );
   FALCON_DECLARE_METHOD( rsplit, "S" );
   FALCON_DECLARE_METHOD( splittr, "S" );
   FALCON_DECLARE_METHOD( startsWith, "S" );
   FALCON_DECLARE_METHOD( wmatch, "S" );
   */
};

}

#endif /* _FALCON_CLASSRE_H_ */

/* end of classre.h */
