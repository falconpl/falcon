/*
  FALCON - The Falcon Programming Language
  FILE: dbi_mod.h
  
  DBI module -- module service classes
  -------------------------------------------------------------------
  Author: Jeremy Cowgar
  Begin: 2007-12-22 10:06
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
 dbi_mod.h - DBI module -- module service classes
*/

#ifndef flc_dbi_mod_H
#define flc_dbi_mod_H

#include <falcon/userdata.h>
#include <falcon/string.h>

#include "dbi_mod.h"

namespace Falcon {

   class DBIConnection;
   class DBIRecordset;

   class DBIConnection : public UserData
   {
   protected:
      int m_errorCode;
      String m_errorMessage;

      int setErrorInfo( const int code, const char *message ) {
         m_errorCode = code;
         m_errorMessage = message;
         return m_errorCode;
      }

   public:
      DBIConnection() {}

      virtual int connect( const String *connString );

      virtual int beginTransaction() {};
      virtual int rollbackTransaction() {};
      virtual int commitTransaction() {};

      virtual int execute( const String *sql ) {};
      virtual DBIRecordset *query( const String *sql ) {};

      virtual int close() {};
   };

   class DBIRecordset : public UserData
   {
   protected:
      int m_errorCode;
      String m_errorMessage;

      int m_affectedRows;
      int m_rowCount;
      int m_columnCount;
      int m_rowIndex;

      DBIConnection *m_connClass;

      int setErrorInfo( const int code, const char *message,
                        bool blank = true )
      {
         m_errorCode = code;
         m_errorMessage = message;
         if ( blank )
         {
            m_affectedRows = -1;
            m_rowCount = 0;
            m_columnCount = 0;
            m_rowIndex = 0;
         }
         return m_errorCode;
      }

   public:
      DBIRecordset( DBIConnection *connClass ) {
         m_connClass = connClass;
      }

      virtual int columnIndex( const String *columnName ) {};
      virtual int columnName( const int columnIndex, String &name ) {};
      virtual int value( const int columnIndex, String &value ) {};

      virtual int next() {};
      virtual int close() {};
   };

}

#endif

/* end of dbi_mod.h */
