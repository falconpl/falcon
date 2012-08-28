/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_error.h

   Database Interface - Error management
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 May 2010 23:47:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_DBI_ERROR_H_
#define FALCON_DBI_ERROR_H_

#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_DBI_ERROR_BASE
   #define FALCON_DBI_ERROR_BASE 2000
#endif

#define FALCON_DBI_ERROR_COLUMN_RANGE     (FALCON_DBI_ERROR_BASE+1)
#define FALCON_DBI_ERROR_INVALID_DRIVER   (FALCON_DBI_ERROR_BASE+2)
#define FALCON_DBI_ERROR_NOMEM            (FALCON_DBI_ERROR_BASE+3)
#define FALCON_DBI_ERROR_CONNPARAMS       (FALCON_DBI_ERROR_BASE+4)
#define FALCON_DBI_ERROR_CONNECT          (FALCON_DBI_ERROR_BASE+5)
#define FALCON_DBI_ERROR_QUERY            (FALCON_DBI_ERROR_BASE+6)
#define FALCON_DBI_ERROR_QUERY_EMPTY      (FALCON_DBI_ERROR_BASE+7)
#define FALCON_DBI_ERROR_OPTPARAMS        (FALCON_DBI_ERROR_BASE+8)
#define FALCON_DBI_ERROR_NO_SUBTRANS      (FALCON_DBI_ERROR_BASE+9)
#define FALCON_DBI_ERROR_NO_MULTITRANS    (FALCON_DBI_ERROR_BASE+10)
#define FALCON_DBI_ERROR_UNPREP_EXEC      (FALCON_DBI_ERROR_BASE+11)
#define FALCON_DBI_ERROR_BIND_SIZE        (FALCON_DBI_ERROR_BASE+12)
#define FALCON_DBI_ERROR_BIND_MIX         (FALCON_DBI_ERROR_BASE+13)
#define FALCON_DBI_ERROR_EXEC             (FALCON_DBI_ERROR_BASE+14)
#define FALCON_DBI_ERROR_FETCH            (FALCON_DBI_ERROR_BASE+15)
#define FALCON_DBI_ERROR_UNHANDLED_TYPE   (FALCON_DBI_ERROR_BASE+16)
#define FALCON_DBI_ERROR_RESET            (FALCON_DBI_ERROR_BASE+17)
#define FALCON_DBI_ERROR_BIND_INTERNAL    (FALCON_DBI_ERROR_BASE+18)
#define FALCON_DBI_ERROR_TRANSACTION      (FALCON_DBI_ERROR_BASE+19)
#define FALCON_DBI_ERROR_CLOSED_STMT      (FALCON_DBI_ERROR_BASE+20)
#define FALCON_DBI_ERROR_CLOSED_RSET      (FALCON_DBI_ERROR_BASE+21)
#define FALCON_DBI_ERROR_CLOSED_DB        (FALCON_DBI_ERROR_BASE+22)
#define FALCON_DBI_ERROR_DB_NOTFOUND      (FALCON_DBI_ERROR_BASE+23)
#define FALCON_DBI_ERROR_CONNECT_CREATE   (FALCON_DBI_ERROR_BASE+24)
#define FALCON_DBI_ERROR_CLOSING	      (FALCON_DBI_ERROR_BASE+25)

namespace Falcon
{

/** Base error class for all DBI errors.

    DBI Error descriptions are available in English ONLY, until
    the new per-module string table support is ready.
 */
class DBIError: public ::Falcon::Error
{
public:
   DBIError():
      Error( "DBIError" )
   {}

   DBIError( const ErrorParam &params  );

private:
   void describeError();
};

}

#endif
