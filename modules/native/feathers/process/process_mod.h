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

#include <falcon/genericvector.h>

namespace Falcon {

class String;
class VMachine;

/** Process and child management API */
namespace Mod {

class Handle;

/**  Tokenizes a command string with its paremters and appends it to a vector.
 * \param argv Command tokens will be appended to it.
 * \param params Command string.
 */
 void argvize(GenericVector& argv, const String &params);

const char *shellName();
const char *shellParam();

}} // ns Falcon::Mod

#endif

/* end of process_mod.h */
