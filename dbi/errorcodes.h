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
      // Everything is OK
      DBI_OK = 0,

      // Something returned a NULL that shouldn't have
      DBI_MEMORY_ALLOC_ERROR,

      // A new connection could not be established
      DBI_CONNECTION_ERROR,

      // An operation was requested that requires a connection, but one
      // doesn't exist
      DBI_NO_CONNECTION_ERROR,

      // The SQL query failed
      DBI_QUERY_ERROR,

      // An operation was requested that requiers a result set, but one
      // doesn't exist
      DBI_NO_RESULT_ERROR,

      // A column index was requested that is < 0 or > column count
      DBI_COLUMN_INDEX_ERROR,

      // A value was asked for that is NULL
      DBI_NULL_VALUE_WARNING,

      // The SQL->next() has reached the end of the result set
      DBI_EOF_WARNING,
   };

}

#endif

/* end of errorcodes.h */
