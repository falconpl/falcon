/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_recordset.h

   Database Interface - SQL Recordset class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_RECORDSET_H_
#define FALCON_DBI_RECORDSET_H_

#include <falcon/falcondata.h>

namespace Falcon {

class DBITransaction;
class Item;

/**
 * Abstraction of recordset class.
 *
 * The recordset class is the minimal query access interface unit towards the database.
 * It represents a single database query with results. Through this class, query data
 * can be accessed.
 */
class DBIRecordset : public FalconData
{

public:
   DBIRecordset( DBITransaction* generator );
   virtual ~DBIRecordset();

   /** Move to the next record
    * \throw DBIError* in case of error.
    * \return true on success, false on end of updates reached
    */
   virtual bool fetchRow()=0;

   /**
    * Get the current row number.
    *
    * \return row index (0 based) or -1 for invalid row
    */
   virtual int64 getRowIndex()=0;

   /**
    * Fetch the number of rows in the recordset or -1 if unknown
    */
   virtual int64 getRowCount()=0;

   /**
    * Fetch the number of columns in the recordset
    */
   virtual int getColumnCount()=0;

   /**
    * Fetch the row headers
    */
   virtual bool getColumnName( int nCol, String& name )=0;

   /** Gets a value in the recordset.
    */
   virtual bool getColumnValue( int nCol, Item& value )=0;

   /** Gets a type in the recordset.
    */
   //virtual dbi_status getColumnType( int nCol, dbi_type& type )=0;

   /** Skip the required amount of records from this position on. */
   virtual bool discard( int64 ncount ) = 0;

   /**
    * Close the recordset
    */
   virtual void close()=0;

   //=========================================================
   // Manage base class control.
   //
   virtual FalconData *clone() const;
   virtual void gcMark( uint32 );

protected:
   DBITransaction* m_trh;
};

}

#endif

/* end of dbi_recorset.h */
