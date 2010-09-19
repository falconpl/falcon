/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_error.h
   $Id$

   Falcon module for Apache 2

   Internal and pre-vm operations error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

//=============================================
// Simple write doc function:
//

#ifndef MOD_FALCON_ERROR_H
#define MOD_FALCON_ERROR_H

#include "mod_falcon.h"

void falcon_mod_write_errorstring( request_rec *rec, const char *str );

#endif
