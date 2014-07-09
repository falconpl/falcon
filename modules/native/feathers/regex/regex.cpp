/*
   FALCON - The Falcon Programming Language.
   FILE: regex.cpp

   Regular expression extension
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
  Regular expression extension
*/

#include "regex_fm.h"

/*#
   @module regex Regular Expression
   @ingroup feathers
   @brief Regular expression based string matching and substitution (PCRE binding).
   
   Regular expressions are a powerful mean to search for patterns in strings and to
   extract substrings matching a certain criterion.

   Falcon uses PCRE, or "PERL Compatible Regular Expression", version 7.6, as
   codebase for the Regex module. Scanning and matching international and wide
   strings is fully supported. Version 7.6 of PCRE offers also extended functionalities
   for some Unicode category, in example it can recognize also Unicode "wide"
   whitespaces. Discussion of usage of regular expression, and specifically, of PERL
   compatible regular expressions is beyond the scope of this guide. Further material on
   the topic can be found at:
   - http://en.wikipedia.org/wiki/Regular_expression
   - http://www.regular-expressions.info

   The original codebase, with complete documentation, is at http://www.pcre.org.

   Basic usage of Falcon Regular Expression module is quite simple. After importing
   the Regex module, one regular expression can be compiled using the Regex() class
   constructor, and then the instance can be used to match repeatedly strings.

   The Falcon Regex module is based on concepts and pattern usages that are meant to
   minimize wastes and inefficiencies that are usually bound with scripting language
   usages. In example, it is quite common to extract substrings from the pattern using
   "parenthesized" expressions, that are actually subpatterns in parenthesis. But as
   parenthesis could be used to simply build recognition patterns, the substring they
   match is not always needed by the calling application. Other times, the calling
   application is more interested in knowing where a substring starts and ends rather
   than in its content. While other scripting languages build a whole set of matched
   substrings as the match happens, Falcon Regex module provides the user with set of
   range items that can be applied on the original string either to know where the
   matches took place or to extract a substring in one single VM step.

A minimal example would look like the following:
@code
   load regex

   re = Regex( "find:\\s*(.*);" )  // Use "\\" to get "\", see below
   string = "this is something to find: the text or 言葉; (and this will be ignored)"

   if re.match( string )
      range = re.captured(1)
      printl( "Match was found at: ", re.captured(0) )
      printl( "Parenthesized capture: >", string[ range ], "<" )
   else
      printl( "No match" )
   end
   @endcode

   The above code will create the re instance, and then will use it to search any text (.*)
   between "find:", eventually followed by any number of whitespaces (\s*), and the
   first ";". The capture method will return the nth closed range that has been matched;
   the "0" range is the complete match, that is the match including "find" and ";". As any
   Falcon range, the match result can be cached in any variable (range in our example)
   and applied to the string later.

   Please, notice the double "\\" in the above Regex() constructor; Falcon string parsing
   would trasform \s into a "silent whitespace", and would parse "\n" as a physical new
   line, \t as a tab and so on. Regular expression pattern may include many of those
   characters, giving them different meanings. For this reason, it is necessary to use
   the "\\" escape where in regular expression documentation (and in user interfaces) a
   single "\" is required. If you receive the string from an external source
   (i.e. a file or a user input), there is no need for this escape; this is just for Falcon
   compiler to produce a correct regular expression.

   The above program will output:
   @code
   Match was found at: [21:42]
   Parenthesized capture: >the text or 言葉<
   @endcode

   The parser provides single quoted literal strings, where the backslash has no special
   meaning.

   In example:
   @code
   re = Regex( '^\s*\w' )
   @endcode

   This would match any word near the beginning of a line, and it is equivalent to:
   @code
   re = Regex( "^\\s*\\w" )
   @endcode

   Of course, the former is more readable and natural when dealing with regular
   expressions.

   @beginmodule regex
*/

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new ::Falcon::Feathers::ModuleRegex;
   return mod;
}

#endif

/* end of regex.cpp */

