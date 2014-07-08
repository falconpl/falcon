/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi.cpp
 *
 * Database interface - Main Module
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun Dec 2007 23 21:54:34 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi/dbi.cpp"

#include "dbi.h"
#include <falcon/dbi_handle.h>
#include <falcon/syntree.h>
#include <falcon/classes/classmodule.h>
#include "dbi_classhandle.h"
#include "dbi_classrecordset.h"
#include "dbi_classstatement.h"

#include <falcon/dbi_error.h>
#include <falcon/dbi_service.h>
#include <falcon/pstep.h>
#include <falcon/vmcontext.h>
#include <falcon/modspace.h>
#include <falcon/stdhandlers.h>

#include "version.h"

/*# @beginmodule  dbi */

namespace Falcon {

namespace {

class PStepCatchSubmoduleLoadError: public SynTree
{
public:
   PStepCatchSubmoduleLoadError() { apply = apply_; setTracedCatch(); }
   virtual ~PStepCatchSubmoduleLoadError() {}
   virtual void describeTo( String& target ) const
   {
      target = "PStepCatchSubmoduleLoadError";
   }


   static void apply_(const PStep*, VMContext* ctx)
   {
      DBIError* error = new DBIError(ErrorParam(FALCON_DBI_ERROR_INVALID_DRIVER, __LINE__, SRC).symbol("connect"));
      error->appendSubError(ctx->thrownError());
      ctx->popCode();
      ctx->raiseError(error);
   }
};

class PStepAfterSubmoduleLoad: public PStep
{
public:
   PStepAfterSubmoduleLoad() { apply = apply_; }
   virtual ~PStepAfterSubmoduleLoad() {}
   virtual void describeTo( String& target ) const
   {
      target = "PStepAfterSubmoduleLoad";
   }


   static void apply_(const PStep*, VMContext* ctx)
   {
      Module* loaded = static_cast<Module*>(ctx->topData().asInst());
      // todo; more robust check.

      DBIService* srv = static_cast<DBIService*>(loaded->createService(FALCON_DBI_HANDLE_SERVICE_NAME));
      if( srv == 0 )
      {
         throw FALCON_SIGN_XERROR(DBIError, FALCON_DBI_ERROR_INVALID_DRIVER, .extra("Given module is nota  DBI driver") );
      }

      String* params = ctx->param(0)->asString();
      uint32 colonPos = params->find( ":" );
      String connString;
      if ( colonPos != csh::npos )
      {
         connString = params->subString( colonPos + 1 );
      }

      // can throw, and it's ok.
      DBIHandle* dbh = srv->connect(connString);
      // same
      if( ctx->paramCount() > 1 )
      {
         dbh->options( *ctx->param(1)->asString() );
      }

      // the proper class handler is provided by the handler itself,
      // and that class handler will keep the module alive.
      ctx->returnFrame(FALCON_GC_HANDLE(dbh));
      // we don't have to free the service
   }
};

/*#
   @function connect
   @brief Connect to a database server through a DBI driver.
   @param conn SQL connection string.
   @optparam queryops Default transaction options to be applied to
                 operations performed on the returned handler returned handle.
   @return an instance of @a Handle.
   @raise DBIError if the connection fails.

  This function acts as a front-end to dynamically determine the DBI driver
  that should be used to connect to a determined database.

  The @b conn connection string is in the format described in @a dbi_load.
  An optional parameter @b queryops can be given to change some default
  value in the connection.

  @see Handle.options
*/
FALCON_DECLARE_FUNCTION(connect, "conn:S,queryops:[S]");
FALCON_DEFINE_FUNCTION_P1(connect)
{
   //static Log* log = Engine::instance()->log();

   Item *paramsI = ctx->param(0);
   Item *i_tropts = ctx->param(1);
   if (  paramsI == 0 || ! paramsI->isString()
         || ( i_tropts != 0 && ! i_tropts->isString() ) )
   {
      throw paramError(__LINE__);
   }

   String *params = paramsI->asString();
   String provName;
   //String connString = "";
   uint32 colonPos = params->find( ":" );

   if ( colonPos != csh::npos )
   {
      provName = "dbi." + params->subString( 0, colonPos );
      //connString = params->subString( colonPos + 1 );
   }
   else {
      provName = "dbi." + *params;
   }

   DBIModule* mod = static_cast<DBIModule*>(module());
   ctx->pushCodeWithUnrollPoint(mod->m_stepCatchSubmoduleLoadError);
   ctx->pushCode(mod->m_stepAfterSubmoduleLoad);

   // Work in our own module workspace
   ModSpace* ms = mod->modSpace();
   // already there?
   Module* driver = ms->findByName(provName);
   if( driver != 0 )
   {
      static Class* modClass = Engine::instance()->stdHandlers()->moduleClass();
      // let our steps to manage the situation.
      ctx->pushData( Item(modClass, driver) );
   }
   else {
      // ask the module space to try loading this module.
      ms->loadModuleInContext( provName, false, false, false, ctx, mod, true);
   }
   // don't return.
}

} // end of nameless namespace

/*#
   @module dbi Falcon Database Interface.
   @after feathers
   @brief Main module for the Falcon DBI module suite.

   Falcon Database Interface (DBI) is a common infrastructure that interfaces
   different SQL database engines with the Falcon Programming Language.

   The DBI module suite presents a base interface module through which all the
   engines can be accessed by the script as a seamless @a Handle instance, where
   the nature of the underlying engine is abstracted and hidden away from the
   final program.

   Other than presenting a common interface, each engine driver is wrapped in
   a separate module that can be loaded or imported directly by the final user,
   and that offers engine-specific functionalities as well as the common DBI 
   interface.

   More precisely, every database driver inherits from the base @a Handle class
   and may offer functionalities that cover more adequately the specific engine
   that they interface.

   @note At version 1.x, all the modules just cover the basic DBI requirements.
         Coverage of engine-specific functionalities is due for version 2.0.

   @section dbi_load DBI foreign modules architecture
   
   The DBI module automatically finds and instantiates the proper subclass of the
   @a Handle object through the @a connect Function. A "database resource locator" (DRL),
   which is a string describing connection parameters, must be provided to the
   @a connect function; the first element of the DRL is a database name, separated
   from the rest of the DRL by a colon, which must mirror the name of a DBI enabled
   module (usually a binary module in a .dll/.so/.dylib file), where the DBI @a Handle
   class is derived. Other parameters of the connection string are provided in 
   "<name>=<value>" format, and separated by semicolons.

   For example, to connect a local @a http://sqlite3.org/ SQLite3 database, the DRL is
   the following:

   @code
   import from dbi
   
   hdb = dbi.connect( "sqlite3:db=/home/user/mydb.db" )
   @endcode

   DRL strings can have driver-specific parameters, but the following parameters are known by
   all the drivers (even if not necessarily used):

   - **db**: The database name or path (depending on the underlying driver model)
   - **uid**: User ID (user name, account or similar).
   - **pwd**: Password. To perform a password-less connection, don't provide this parameter.
             To send an empty password, pass the value as "pwd=;"
   - **host**: Host where to connect, in case the database engine is network based. When not
              given, it defaults to "nothing", or in other words, the target engine default
              is picked.
   - **port**: TCP Port where to connect, in case the database engine is network based. It is passed
              untraslated to the engine, so it must respect the values that the engine may accept.
              If not given, the default setting for the required engine is used.

   @section dbi_settings Common DBI settings

   As a database abstaction layer, DBI tries to enforce a set of policies that stay common across
   different database engines, causing different engines with different defaults to behave coherently.
   If the database is not able to support a certain default behavior that DBI tries to apply to it,
   then the default is ignored. In case the DBI user tries to explicitly enforce a non-supported
   policy by setting the @a dbi_opts "connection options", then a @a DBIError is raised.

   DBI options can be set at connection through a second parameter or during the rest of the work
   through the @a Handle.options method. Options can be set through a string whose content is
   in the same format of the DRI (<parameter>=<value>;...)

   In general, connections are opened with any kind of autocommit mechanism turned off. An explicit @a Handle.commit 
   must issued if the operations perfomred on the database are meant to be actually saved; closing the database
   or exiting from the virtual machine in an unclean way causes an implicit rollback of any pending change.
   Autocommit can be turned on through the "autocommit=on" option.

   Queries are fully fetched client-side before the control is returned to the script. Not all the engines
   can change this behavior, but the "prefetch" option can be used to keep the resultset in the server
   (prefetch=none) or to transfer the resultset only if it is smaller than a given amount (prefetch=N). In this
   second case, the DBI module will try to transfer N records at a time, if possible.

   @section dbi_methods DBI common methods and usage patterns

   As a database layer abstraction, DBI tries to perform operations differently handled under
   different engines in a way that can be considered valid across the widest range of databases possible.

   DBI distinguishes between SQL queries (operations meant to return recordsets, as, for example, "select"), 
   SQL statements not meant to return any recordset and stored procedure calls, which may or may not return
   a recordset. In the DBI framework, prepared statements cannot be seen as queries; when the engine supports
   queries as prepared statements, this is internally managed by the DBI engine. 

   To perform queries, the user should call the @a Handle.query method, that will return a @a Recordset instance
   (even if empty), nil if the query didn't return any recordset.

   If the stored procedure can produce resultset even when not being invoked from an explicit "select"
   statement (that may be invoked by a falcon), @a Handle.call may return a @a Recordset instance. It will
   return @b nil if this is not the case.

   Finally, the @a Handle.prepare method returns an instance of the @a Statement class, on which multiple
   @a Statement.execute can be invoked. As different database engine behave very differently on this regard,
   the base @a Statement.execute method never returns any recordset.

   @subsection dbi_pos_params Positional parameters

   All the four SQL command methods (query and prepare) accept positional parameters as question
   mark placeholders; the SQL parameters can be then specificed past the first parameter (except in the case of
   the @a Handle.prepare, where the parameters are to be specified in through the returned @a Statement class). 
   If the underlying engine supports positional parameters, then DBI uses the database driver specific functions
   to pass the parameters directly to the engine, otherwise the parameters are transformed into SQL aware string
   values and inserted in the statement prior passing it to the underlying functions. In case of engines
   using different conventions for positional parameters, while the engine-specific convention can be used,
   the engine ensures that the question mark can be used as a positional parameter indicator, and eventually
   turns it into the engine-specific placeholder prior passing it to the underlying functions.

   When the parameters are stored in a Falcon array, the array call semantic can be used as in the following
   example:

   @code
   import from dbi

   dbh = dbi.connect( "..." )
   
   limits = [ 123, "2010-3-1", "2010-4-1"]

   // select all the data in the limits
   dbr = (.[ dbh.query "
         select * from mytab 
            where 
               f1 < ? and 
               date_start >= ? and
               date_end <= ?" ] + limits)()

   @endcode

   The above code is semantically equivalent to calling the @ dbh.query method applying the parameters
   specified in the @b limits array.

   @subsection dbi_fetching Fetching data from queries

   Fetch is the basic operation that allows to retreive data from queries. Every succesful query or stored
   procedure call returning data generates a @a Record instance that can be iteratively fetched for all the
   data to be returned.

   The @a Recordset.fetch method accepts a value where the incoming query data will be stored. Recycling the
   same input parameter allows to minimize the memory requirements that a script needs to fetch a recordset.

   The following example is the most common pattern to retreive all the data stored in a table:

   @code
   import from dbi

   dbh = dbi.connect( "..." )
   dbr = dbh.query( "select * from mytable" )

   row = []
   while dbr.fetch( row )
      > row.describe()
   end
   @endcode

   A @a Recordset instance provides method to retreive informations about the size of the recordset, the
   list of the retreived colums, their type and so on. 

   In case the structure of the table is unknwon, it may be useful to invoke @a Recordset.fetch passing an
   empty dictionary; this will fill the dictionary with column-name => column-value pairs:

   @code
   import from dbi

   dbh = dbi.connect( "..." )
   dbr = dbh.query( "select * from mytable" )
   
   > "Data in the first row: "
   inspect( dbr.fetch( [=>] )
   @endcode

   The @a Recordset.fetch method can also accept a Table instance; if not yet configured, the table will
   get shaped on the colums retrieved from the query, and will be filled with all the data from the
   recordset in one step:

   @code
   import from dbi

   dbh = dbi.connect( "..." )
   dbr = dbh.query( "select * from mytable" )
   
   // dbr.fetch returns the same object passed as the parameter
   // the [3] accessor gets the fourth row, which is an array
   > "Data in the fourth row: " + dbr.fetch( Table() )[3].describe()
   @endcode

   @section dbi_value_types Database and Falcon values
   
   Databases have different, often non-standard data-types that must be mapped into Falcon item types
   when they must be translated into SQL parameters or when they are returned as SQL query results.
   
   DBI transform database data types into falcon items, and eventually into Falcon standard language
   class instances, using the nearest data type. The following list indicates how data coming from
   and going to the database is transformed in/to Falcon item types.
   
   - @b nil: This values indicates a NULL database value.
   - @b integer: Falcon 64-bit integers are used for all the integer-based operations. When they
                 are used as parameters to be stored in fields with smaller precision, the values
		 may be truncated.
   - @b numeric: Falcon 64-bit floating point values are used to store REAL, FLOAT and DOUBLE data types,
                 as well as fixed-point data types as DECIMAL or FIXED. If the precision of the underlying
		 database type is smaller than the IEEE 64 bit floating point variable type (a C double
		 data type), then the value may be truncated or may lose precision. If it has a greater
		 precision than the Falcon data type, then it's the Falcon output variable the one that
		 may get truncated. In that case, consider using the "string" parameter for the database
		 connection (see below).
    - @b string: Text based fields as fixed or variable length CHAR fields or text-oriented blobs can be
                 updated using string values and are saved in output to Falcon strings. If the database
		 support text encoding, then the transforamation between Falcon strings and underlying
		 encoding is transparent (in both the directions).  If the encoding is not supported, or
		 the field is a binary-oriented CHAR or encoding neutral, then the data retrieved in the
		 string must be transcoded by the Falcon application with the usual means (using a transcoded
		 string stream or using the transcodeTo/transcodeFrom string methods).
    - @b MemBuf: Falcon memory buffers can be used to store binary data into binary blobs, and are created
                 when reading binary blob fields from the database. Usually, they can also be sent to
                text fields, in which case they will be stored in the database as binary sequences without
                any database conversion. To create a memory buffer, use a mutable string and set its
                isText field to false.
    - @b TimeStamp: Falcon TimeStamp standard class instances can be used to store date and time related
                 datatype, and they are created when retrieving this kind of fields. Timezones
		 are ignored, and they are not restored when reading informations from the database.
		 When storing values into database fields that reprsent time-related values but provide just
		 part of the information reported by TimeStamp, extra data is discarded. When reading
		 fields containing only partial time information, unexpressed data is zeroed. So, when
		 reading a DATE sql field, the hour, minute, second and millisecond fields of the resulting
		 TimeStamp instance are zeroed.
     - @b Object: When presenting any other object as an input field in a SQL statement, the Object.toString
                 method is applied and the result is sent to the database driver instead.

   The "string=on" option can be specified in the connection parameter or database handle option (see @a Handle.options) 
   to have
   all the results of the queries returned as string values, except for NULL and binary blobs, that are still
   returned as @b nil and @b MemBuf items. If the underlying engine supports this method
   natively and the extracted data should just be represented on output, or if the database engine provides
   some information that cannot be easily determined after the automatic Value-to-Item translation, this
   modality may be extremely useful.

   @beginmodule dbi
*/


// the main module
DBIModule::DBIModule():
   Module("dbi")
{
   m_stepCatchSubmoduleLoadError = new PStepCatchSubmoduleLoadError;
   m_stepAfterSubmoduleLoad = new PStepAfterSubmoduleLoad;

   m_handleClass = new DBI::ClassHandle;
   m_statementClass = new DBI::ClassStatement;
   m_recordsetClass = new DBI::ClassRecordset;

   *this
      << new Function_connect
      << m_handleClass
      << m_statementClass
      << m_recordsetClass
      << new ClassDBIError
      ;
}

DBIModule::~DBIModule()
{
   delete m_stepCatchSubmoduleLoadError;
   delete m_stepAfterSubmoduleLoad;
}

}


FALCON_MODULE_DECL
{
    return new Falcon::DBIModule;
}

/* end of dbi.cpp */

