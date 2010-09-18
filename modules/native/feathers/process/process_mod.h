/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.h

   Process API definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process API definitions
*/

#ifndef flc_process_mod_H
#define flc_process_mod_H

namespace Falcon {

class String;
class VMachine;

/** Process and child management API */
namespace Mod {

class Handle;

/** Returns an array of char * where each element points to a token in "params".
   This function creates (using Falcon::memAlloc()) a vector of char *; each element
   of this vector points to the beginning of a token in the params string. That
   string must be formatted so that each parameter is separated by the
   previous one by a '\0'; the size parameter counts how many tokens have to
   be indexed in the string. The extra parameter indicates how many elements
   other than 'size' must be added to the returned array, so that if i.e.
   size is 4 and extra is 2, the returned array will have 6 elements. Start
   parameter indicates the position where the first token pointer must be
   placed; i.e. if you want to put something that is not in params at the
   first place in the returned array, put 1 as start, so that the element 0
   will be at your disposal.
   \note This function tipically post-parses data returned by parametrize().
   \param params the string to be parsed
   \param addShell true to add a shell wrapper.
   \return a newly allocated vector
*/
String **argvize( const String &params, bool addShell );


/** Splits a string in substrings in a row.
   The input is splitted into substrings by changing the first blank
   after each entry in a '\0'; blanks after the first blank are
   removed, so that a input string like
   \code
   "c1  c2 c3\t  c4\0"
   \endcode
   becomes
   \code
   "c1\0c2\0c3\0c4\0"
   \endcode

   @param in the string to be tokenized
   @param out the destination buffer. It must be preallocated to a size that
      can contain the in string (out string will be long as in or less).
   \return the count of tokens in the string, so a parsing function can
   thereafter understand how many strings are counted for. A minimal token
   count is 1 (the whole in string); count can be 0 if the string is empty.
*/
//int parametrize( char *out, const String &in );

/** Frees the string pointer used to create the parameter array */
void freeArgv( String **argv );

const char *shellName();
const char *shellParam();

}

}

#endif

/* end of process_mod.h */
