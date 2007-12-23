/*
  FALCON - The Falcon Programming Language
  FILE: errorcodes.h
  
  Common DBI error codes
  -------------------------------------------------------------------
  Author: Jeremy Cowgar
  Begin: 2007-12-22 18:39
  Last modified because:
  
  -------------------------------------------------------------------
  (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
  
  See LICENSE file for licensing details.
  In order to use this file in its compiled form, this source or
  part of it you have to read, understand and accept the conditions
  that are stated in the LICENSE file that comes boundled with this
  package.
*/

/** \file
	 errorcodes.h - Common DBI error codes
*/

#ifndef flc_dbi_errorcodes_H
#define flc_dbi_errorcodes_H

namespace Falcon
{

   enum {
      DBI_OK = 0,
      DBI_MEMORY_ALLOC_ERROR,
      DBI_NO_CONNECTION_ERROR,
      DBI_QUERY_ERROR,
      DBI_NO_RESULT_ERROR,
      DBI_NULL_VALUE_WARNING,
      DBI_EOF_WARNING,
   };

}

#endif

/* end of errorcodes.h */
