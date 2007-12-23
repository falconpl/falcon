/*
  FALCON - The Falcon Programming Language
  FILE: sql_mod.h
  
  SQL module -- module service classes
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
	 sql_mod.h - SQL module -- module service classes
*/

#ifndef flc_sql_mod_H
#define flc_sql_mod_H

#include <falcon/string.h>

namespace Falcon {
	
	enum {
		SQL_OK = 0,
	};
	
	class SQLConnection;
	class SQLRecordset;
	
	class SQLConnection : public UserData
	{
	public:
		SQLConnection(const String *connString) {};

		virtual int beginTransaction() {};
		virtual int rollbackTransaction() {};
		virtual int commitTransaction() {};

		virtual int execute(const String *sql) {};
		virtual SQLRecordset *query(const String *sql) {};

		virtual int close() {};
	};
	
	class SQLRecordset : public UserData
	{
	private:
		SQLConnection *m_connClass;

	public:
		SQLRecordset(SQLConnection *connClass);

		virtual int columnIndex(const String *columnName);
		virtual int columnName(int columnIndex, String &name);
		virtual int value(int columnIndex, String &value);

		virtual int next();
		virtual int close();
	};

}

#endif

/* end of sql_mod.h */
