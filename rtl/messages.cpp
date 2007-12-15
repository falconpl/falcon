/*
   FALCON - The Falcon Programming Language
   FILE: messages.cpp
   $Id: messages.cpp,v 1.1 2007/02/24 22:03:34 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab feb 24 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Short description
*/

#include <falcon/setup.h>
#include "falcon_rtl_ext.h"

namespace Falcon {
namespace Ext {

wchar_t *message_table[] = {
   L"required an array, a start and an end position",
   L"invalid start/end positions",
   L"requres an array and another parameter",
   L"optional third parameter must be a number",
   L"optional fourth parameter must be a number",
   L"scan end is greater than start",
   L"requires an array as first parameter",
   L"second parameter must be callable",
   L"needs two arrays",
   L"indexes must be numbers",
   L"start position out of range",   // 10
   L"parameter array contains non string elements",
   L"parameter array is empty",
   L"Iterator class not found in VM",
   L"Given item is not a valid iterator for the collection",
   L"Sender is not an object",
   L"Marshalled event name must be a string as first element in the given array",
   0
   };

}}


/* end of messages.cpp */
