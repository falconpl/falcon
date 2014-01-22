/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_classrecordset.cpp
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Tue, 21 Jan 2014 16:38:11 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi/dbi_classrecordset.cpp"

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/itemarray.h>
#include <falcon/itemdict.h>

#include "dbi.h"
#include "dbi_mod.h"
#include "dbi_classrecordset.h"

#include <falcon/dbi_recordset.h>

/*# @beginmodule dbi */

namespace Falcon {

namespace {

   class PStepStoreFetchedProperties: public PStep
   {
   public:
      PStepStoreFetchedProperties() { apply = apply_; }
      virtual ~PStepStoreFetchedProperties() {}
      virtual void describeTo( String& target ) const { target = "PStepStoreFetchedProperties"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         DBIRecordset *dbr = ctx->tself<DBIRecordset *>();
         int32& seqId = ctx->currentCode().m_seqId;
         long depth = ctx->codeDepth();

         // get the fetchable parameter -- we know it's param 1
         Class* cls = 0;
         void* inst = 0;
         ctx->param(1)->asClassInst(cls, inst);
         Item value;

         // all column fetched?
         while( seqId < dbr->getColumnCount() )
         {
            // fetch next column name and value
            String propName;
            dbr->getColumnName(seqId,propName);
            dbr->getColumnValue(seqId,value);
            // prepare to abandon frame
            ++seqId;
            cls->op_setProperty(ctx, inst, propName);
            // went deep?
            if( ctx->codeDepth() > depth )
            {
               break;
            }
         }

         // all complete.
         ctx->popCode();
      }
   };

   static void internal_record_fetch( VMContext* ctx, DBIRecordset* dbr, Item& target )
   {
      int count = dbr->getColumnCount();

      if( target.isArray() )
      {
         ItemArray& aret = *target.asArray();
         aret.resize( count );
         for ( int i = 0; i < count; i++ )
         {
            dbr->getColumnValue( i, aret[i] );
         }
         ctx->popCode();
      }
      else if( target.isDict() )
      {
         ItemDict* dret = target.asDict();
         for ( int i = 0; i < count; i++ )
         {
            String fieldName;
            dbr->getColumnName( i, fieldName );
            Item* value = dret->find( Item(fieldName.handler(), &fieldName) );
            if( value == 0 )
            {
               Item v;
               dbr->getColumnValue( i, v );
               String* key = new String(fieldName);
               key->bufferize();
               dret->insert( FALCON_GC_HANDLE(key), v );
            }
            else
            {
               dbr->getColumnValue( i, *value );
            }
         }
         ctx->popCode();
      }
      else {
         ClassRecordset* crs = static_cast<ClassRecordset*>(ctx->self().asClass());
         ctx->stepIn(crs->m_stepStoreFetchedProperties);
      }
   }

   /*#
      @method fetch Recordset
      @brief Fetches a record and advances to the next.
      @optparam item Where to store the fetched record.
      @raise DBIError if the database engine reports an error.
      @return The @b item passed as a paramter filled with fetched data or
         nil when the recordset is terminated.

      If @b item is not given, a newly created array is filled with the
      fetched data and then returned, otherwise, the given item is filled
      with the data from the fetch.

      The @b item may be:
      - An Array.
      - A Dictionary.
   */

   FALCON_DECLARE_FUNCTION(fetch, "item:[A|D|X]");
   FALCON_DEFINE_FUNCTION_P1(fetch)
   {
      Item *i_data = ctx->param( 0 );

      // prepare the array in case of need.
      if( i_data == 0 )
      {
         ctx->addLocals(1);
         i_data = ctx->local(0);
         *i_data = FALCON_GC_HANDLE(new ItemArray());
         // if i_data is zero, then i_count is also zero, so we don't have to worry
         // about the VM stack being moved.
      }

      if ( ! ( i_data->isArray() || i_data->isDict() || i_data->isUser() ) )
      {
         throw paramError(__LINE__);
      }

      DBIRecordset *dbr = ctx->tself<DBIRecordset *>();

      if( ! dbr->fetchRow() )
      {
         ctx->returnFrame();
      }
      else {
         internal_record_fetch( ctx, dbr, *i_data );
         ctx->returnFrame(*i_data);
      }
   }


   /*#
       @method discard Recordset
       @brief Discards one or more records in the recordset.
       @param count The number of records to be skipped.
       @return true if successful, false if the recordset is over.

       This skips the next @b count records.
   */

   FALCON_DECLARE_FUNCTION(discard, "count:N");
   FALCON_DEFINE_FUNCTION_P1(discard)
   {
      Item *i_count = ctx->param( 0 );
      if ( i_count == 0 || ! i_count->isOrdinal() ) {
         throw paramError();
      }

      DBIRecordset *dbr = ctx->tself<DBIRecordset *>();
      ctx->returnFrame(Item().setBoolean( dbr->discard( i_count->forceInteger() )) );
   }


   /*#
    @property columns DBIRecordset
    @brief Gets the names of the recordset columns.
    This property yields an array of one or more strings, containing the
       names of the rows in the recordset.
   */
   void get_columns(const Class*, const String&, void* instance, Item& target)
   {
      DBIRecordset *dbr = static_cast<DBIRecordset *>( instance );

      int count = dbr->getColumnCount();
      ItemArray* ret = new ItemArray( count );

      for( int i = 0; i < count; ++i )
      {
         String* str = new String;
         dbr->getColumnName( i, *str );
         str->bufferize();
         ret->append( FALCON_GC_HANDLE(str) );
      }

      target = FALCON_GC_HANDLE(ret);
   }

   /*#
      @property currentRow DBIRecordset
      @brief Returns the number of the current row.

       This property yields how many rows have been fetched before the current
       one. It will be -1 if the method @a Recordset.fetch has still not been called,
       0 if the current row is the first one, 1 for the second and so on.

   */
   void get_currentRow(const Class*, const String&, void* instance, Item& target)
   {
      DBIRecordset *dbr = static_cast<DBIRecordset *>( instance );
      target = dbr->getRowCount();
   }

   /*#
      @method getRowCount DBIRecordset
      @brief Return the number of rows in the recordset.
      @return  An integer number >= 0 if the number of the row count row is known,
       -1 if the driver can't access this information.
    */

   void get_rowCount(const Class*, const String&, void* instance, Item& target)
   {
      DBIRecordset *dbr = static_cast<DBIRecordset *>( instance );
      target = dbr->getRowCount();
   }

   /*#
    @property columnCount DBIRecordset
    @brief Return the number of columns in the recordset.
    */

   void get_columnCount(const Class*, const String&, void* instance, Item& target)
   {
      DBIRecordset *dbr = static_cast<DBIRecordset *>( instance );
      target = dbr->getRowIndex();
   }


   /*#
    @method close DBIRecordset
    @brief Close a recordset
    */
   FALCON_DECLARE_FUNCTION(close, "");
   FALCON_DEFINE_FUNCTION_P1(close)
   {
      DBIRecordset *dbr = ctx->tself<DBIRecordset *>();
      dbr->close();
      ctx->returnFrame();
   }

   class PStepDoNext: public PStep
   {
   public:
      PStepDoNext() { apply = apply_; }
      virtual ~PStepDoNext() {}
      virtual void describeTo( String& target ) const { target = "PStepDoNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         DBIRecordset *dbr = ctx->tself<DBIRecordset *>();

         if( ! dbr->fetchRow() )
         {
            ctx->returnFrame();
            return;
         }

         // copy, as we may disrupt the stack
         Item i_callable = *ctx->param(0);
         ctx->pushData(i_callable);

         if( ctx->paramCount() == 1 )
         {
            int count = dbr->getColumnCount();
            for ( int i = 0; i < count; i++ )
            {
               Item value;
               dbr->getColumnValue( i, value );
               ctx->pushData( value );
            }

            ctx->callInternal( i_callable, count );
         }
         else
         {
            internal_record_fetch( ctx, dbr, *ctx->param(1) );
            ctx->pushData(*ctx->param(1));
            ctx->callInternal( i_callable, 1 );
         }
         // call me back!
      }
   };

   /*#
      @method do Recordset
      @brief Calls back a function for each row in the recordset.
      @param cb The callback function that must be called for each row.
      @optparam item A fetchable item that will be filled and then passed to @b cb.
      @raise DBIError if the database engine reports an error.

      This method calls back a given @b cb callable item fetching one row at a time
      from the recordset, and then passing the data to @b cb either as parameters or
      as a single item.

      If @b item is not given, all the field values in the recordset are passed
      directly as parameters of the given @b cb function. If it is given, then
      that @b item is filled along the rules of @b Recordset.fetch and then
      it is passed to the @b cb item.

      The @b item may be:
      - An Array.
      - A Dictionary.

      The @b cb method may return an oob(0) value to interrupt the processing of the
      recordset.

      @b The recordset is not rewinded before starting to call @b cb. Any previously
      fetched data won't be passed to @b cb.
   */

   FALCON_DECLARE_FUNCTION(do, "cb:C,item:[A|D]");
   FALCON_DEFINE_FUNCTION_P1(do)
   {
      Item* i_callable = ctx->param(0);
      Item* i_extra = ctx->param(1);
      if( i_callable == 0 || ! i_callable->isCallable()
          || ( i_extra != 0
               && ! ( i_extra->isArray() || i_extra->isDict() )
               )
        )
      {
         throw paramError(__LINE__);
      }

      ClassRecordset* crec = static_cast<ClassRecordset*>(this->methodOf());
      ctx->stepIn( crec->m_stepDoNext );
   }

   /*#
      @method next Recordset
      @brief Gets the next recordset in queries returning multiple recordsets.
      @return Another recordset on success, nil if this query didn't generate any more
              recordset.
      @raise DBIError if the database engine reports an error.

      Some engines may cause multiple recordsets to be generated after a single query.
      In that case, this method may be used to retrieve the secondary recordest after the
      first one has been completed.

      When there aren't any more recordset to be fetched, this method returns nil.

      The rules to access sub-recorsets may vary depending on the final engine, but
      usually this can be considered safe:

      @code
      rs = dbi.query( "..." )

      // process the query result as usual
      data = [=>]
      while rs.fetch(data)
         // do things
      end

      // process the sub-queries.
      while (sub_rs = rs.next() )
         data = [=>]
         while sub_rs.fetch(data)
            // do more things things
         end
      end

      rs.close()
      @endcode
   */

   FALCON_DECLARE_FUNCTION(next, "");
   FALCON_DEFINE_FUNCTION_P1(next)
   {
      DBIRecordset *dbr = ctx->tself<DBIRecordset *>();

      DBIRecordset *res = dbr->getNext();
      if( res != 0 )
      {
         ctx->returnFrame( FALCON_GC_STORE(methodOf(), res) );
      }
      else {
         ctx->returnFrame();
      }
   }
}

//========================================================================
// Class satatement
//========================================================================

/*#
   @class Recordset
   @brief Data retuned by SQL queries.

   The recordset class abstracts a set of data returned by SQL queries.

   Data can be fetched row by row into Falcon arrays or dictionaries by
   the @a Recordset.fetch method. In the first case, the value extracted
   from each column is returned in the corresponding position of the
   returned array (the first column value at array position [0], the second
   column value in the array [1] and so on).

   When fetching a dictionary, it will be filled with column names and values
   respectively as the key corresponding value entries.

   The @a Recordset.fetch method can also be used to retrieve all the recordset
   contents (or part of it) into a Falcon Table.

   Returned values can be of various falcon item types or classes; see the
   @a dbi_value_types section for further details.

   Other than fetching data, the @a Recordset class can be used to retrieve general
   informations about the recordset (as the returned column size and names).

   @note Closing the database handle while the recordset is still open and in use
   may lead to various kind of errors. It's a thing that should be generally avoided.
*/

ClassRecordset::ClassRecordset():
      Class("%Recordset")
{
   m_stepDoNext = new PStepDoNext;
   m_stepStoreFetchedProperties = new PStepStoreFetchedProperties;

   addMethod( new FALCON_FUNCTION_NAME(fetch) );
   addMethod( new FALCON_FUNCTION_NAME(discard) );
   addMethod( new FALCON_FUNCTION_NAME(close) );
   addMethod( new FALCON_FUNCTION_NAME(do) );
   addMethod( new FALCON_FUNCTION_NAME(next) );

   addProperty( "colums", &get_columns );
   addProperty( "currentRow", &get_currentRow );
   addProperty( "rowCount", &get_rowCount );
   addProperty( "columnCount", &get_columnCount );
}

ClassRecordset::~ClassRecordset()
{
   delete m_stepDoNext;
   delete m_stepStoreFetchedProperties;
}

void ClassRecordset::dispose( void* instance ) const
{
   DBIRecordset* dbh = static_cast<DBIRecordset*>(instance);
   delete dbh;
}

void* ClassRecordset::clone( void* ) const
{
   return 0;
}

void* ClassRecordset::createInstance() const
{
   return 0;
}

void ClassRecordset::gcMarkInstance( void* instance, uint32 mark ) const
{
   DBIRecordset* dbs = static_cast<DBIRecordset*>(instance);
   dbs->gcMark(mark);
}

bool ClassRecordset::gcCheckInstance( void* instance, uint32 mark ) const
{
   DBIRecordset* dbs = static_cast<DBIRecordset*>(instance);
   return dbs->currentMark() >= mark;
}

}

/* end of dbi_classrecordset.cpp */
