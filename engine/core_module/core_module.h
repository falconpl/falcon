/*
   FALCON - The Falcon Programming Language.
   FILE: core_module.h

   Header for Falcon Core Module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:37:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_MODULE_H
#define FALCON_CORE_MODULE_H

#include <falcon/engine.h>
#include <falcon/filestat.h>

namespace Falcon {

class CoreObject;
class Error;

/* Utility function to generate RTL oriented errors */
Error *rtlError( int t, const String &desc );

namespace core {

// Iterator class
FALCON_FUNC  Iterator_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_hasCurrent( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_hasNext( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_hasNext( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_hasPrev( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_next( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_prev( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_value( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_key( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_equal( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_clone( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_erase( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_find( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_insert( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_getOrigin( ::Falcon::VMachine *vm );

FALCON_FUNC  Error_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Error_toString ( ::Falcon::VMachine *vm );
FALCON_FUNC  Error_heading ( ::Falcon::VMachine *vm );
FALCON_FUNC  Error_getSysErrDesc ( ::Falcon::VMachine *vm );
FALCON_FUNC  SyntaxError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  GenericError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  CodeError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  IoError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  TypeError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  AccessError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  MathError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  ParamError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  ParseError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  CloneError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  IntrruptedError_init ( ::Falcon::VMachine *vm );

FALCON_FUNC  len ( ::Falcon::VMachine *vm );
FALCON_FUNC  val_int ( ::Falcon::VMachine *vm );
FALCON_FUNC  val_numeric ( ::Falcon::VMachine *vm );
FALCON_FUNC  typeOf ( ::Falcon::VMachine *vm );
FALCON_FUNC  isCallable ( ::Falcon::VMachine *vm );
FALCON_FUNC  getProperty( ::Falcon::VMachine *vm );
FALCON_FUNC  setProperty( ::Falcon::VMachine *vm );
FALCON_FUNC  chr ( ::Falcon::VMachine *vm );
FALCON_FUNC  ord ( ::Falcon::VMachine *vm );
FALCON_FUNC  hToString ( ::Falcon::VMachine *vm );

FALCON_FUNC  paramCount ( ::Falcon::VMachine *vm );
FALCON_FUNC  _parameter ( ::Falcon::VMachine *vm );
FALCON_FUNC  paramIsRef ( ::Falcon::VMachine *vm );
FALCON_FUNC  paramSet ( ::Falcon::VMachine *vm );

FALCON_FUNC  yield ( ::Falcon::VMachine *vm );
FALCON_FUNC  yieldOut ( ::Falcon::VMachine *vm );
FALCON_FUNC  _f_sleep ( ::Falcon::VMachine *vm );
FALCON_FUNC  beginCritical ( ::Falcon::VMachine *vm );
FALCON_FUNC  endCritical ( ::Falcon::VMachine *vm );
FALCON_FUNC  Semaphore_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Semaphore_post ( ::Falcon::VMachine *vm );
FALCON_FUNC  Semaphore_wait ( ::Falcon::VMachine *vm );
FALCON_FUNC vmSuspend( ::Falcon::VMachine *vm );

FALCON_FUNC  Format_parse ( ::Falcon::VMachine *vm );
FALCON_FUNC  Format_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Format_format ( ::Falcon::VMachine *vm );
FALCON_FUNC  Format_toString ( ::Falcon::VMachine *vm );

FALCON_FUNC  attributeByName( ::Falcon::VMachine *vm );
FALCON_FUNC  having( ::Falcon::VMachine *vm );
FALCON_FUNC  testAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC  giveTo( ::Falcon::VMachine *vm );
FALCON_FUNC  removeFrom( ::Falcon::VMachine *vm );
FALCON_FUNC  removeFromAll( ::Falcon::VMachine *vm );
FALCON_FUNC broadcast_next_attrib( ::Falcon::VMachine *vm );
FALCON_FUNC  broadcast( ::Falcon::VMachine *vm );

FALCON_FUNC  core_exit ( ::Falcon::VMachine *vm );
FALCON_FUNC  PageDict( ::Falcon::VMachine *vm );
FALCON_FUNC Make_MemBuf( ::Falcon::VMachine *vm );

FALCON_FUNC  core_any ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_all ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_anyp ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_allp ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_eval ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_valof ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_min ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_max ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_map ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_dolist ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_times ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_xtimes ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_xmap ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_filter ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_reduce ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_iff ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_choice ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_lit ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_cascade ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_floop ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_firstof ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_let ( ::Falcon::VMachine *vm );

FALCON_FUNC  core_oob( ::Falcon::VMachine *vm );
FALCON_FUNC  core_deoob( ::Falcon::VMachine *vm );
FALCON_FUNC  core_isoob( ::Falcon::VMachine *vm );

FALCON_FUNC  core_lbind( ::Falcon::VMachine *vm );

FALCON_FUNC  gcEnable( ::Falcon::VMachine *vm );
FALCON_FUNC  gcSetThreshold( ::Falcon::VMachine *vm );
FALCON_FUNC  gcSetTimeout( ::Falcon::VMachine *vm );
FALCON_FUNC  gcPerform( ::Falcon::VMachine *vm );
FALCON_FUNC  gcGetParams( ::Falcon::VMachine *vm );

FALCON_FUNC  vmVersionInfo( ::Falcon::VMachine *vm );
FALCON_FUNC  vmModuleVersionInfo( ::Falcon::VMachine *vm );
FALCON_FUNC  vmVersionName( ::Falcon::VMachine *vm );
FALCON_FUNC  vmSystemType( ::Falcon::VMachine *vm );
FALCON_FUNC  vmIsMain( ::Falcon::VMachine *vm );
FALCON_FUNC  vmFalconPath( ::Falcon::VMachine *vm );

FALCON_FUNC  print ( ::Falcon::VMachine *vm );
FALCON_FUNC  printl ( ::Falcon::VMachine *vm );
FALCON_FUNC  inspect ( ::Falcon::VMachine *vm );
FALCON_FUNC  inspectShort ( ::Falcon::VMachine *vm );
FALCON_FUNC  seconds ( ::Falcon::VMachine *vm );
FALCON_FUNC  input ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_getenv( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_setenv( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_unsetenv( ::Falcon::VMachine *vm );
FALCON_FUNC  InputStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  OutputStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  IOStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  systemErrorDescription ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_close ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_flush ( ::Falcon::VMachine *vm );
FALCON_FUNC  StdStream_close ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_read ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readLine ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readText ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_write ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_writeText ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_setEncoding ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_clone ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readItem ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_writeItem ( ::Falcon::VMachine *vm );

#define   CR_TO_CR 0
#define   CR_TO_CRLF 1
#define   SYSTEM_DETECT -1

FALCON_FUNC  Stream_seek ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_seekEnd ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_seekCur ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_tell ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_truncate ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_lastError ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_lastMoved ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_errorDescription ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_eof ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_isOpen ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_writeAvailable ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readAvailable ( ::Falcon::VMachine *vm );

FALCON_FUNC  _stdIn ( ::Falcon::VMachine *vm );
FALCON_FUNC  _stdOut ( ::Falcon::VMachine *vm );
FALCON_FUNC  _stdErr ( ::Falcon::VMachine *vm );

FALCON_FUNC  stdInRaw ( ::Falcon::VMachine *vm );
FALCON_FUNC  stdOutRaw ( ::Falcon::VMachine *vm );
FALCON_FUNC  stdErrRaw ( ::Falcon::VMachine *vm );

FALCON_FUNC  StringStream_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  StringStream_getString ( ::Falcon::VMachine *vm );
FALCON_FUNC  StringStream_closeToString ( ::Falcon::VMachine *vm );


FALCON_FUNC  strSplit ( ::Falcon::VMachine *vm );
FALCON_FUNC  strSplitTrimmed ( ::Falcon::VMachine *vm );
FALCON_FUNC  strMerge ( ::Falcon::VMachine *vm );
FALCON_FUNC  strFind ( ::Falcon::VMachine *vm );
FALCON_FUNC  strBackFind ( ::Falcon::VMachine *vm );
FALCON_FUNC  strFront ( ::Falcon::VMachine *vm );
FALCON_FUNC  strBack ( ::Falcon::VMachine *vm );
FALCON_FUNC  strTrim ( ::Falcon::VMachine *vm );
FALCON_FUNC  strFrontTrim ( ::Falcon::VMachine *vm );
FALCON_FUNC  strAllTrim ( ::Falcon::VMachine *vm );
FALCON_FUNC  strReplace ( ::Falcon::VMachine *vm );
FALCON_FUNC  strReplicate ( ::Falcon::VMachine *vm );
FALCON_FUNC  strBuffer ( ::Falcon::VMachine *vm );
FALCON_FUNC  strUpper ( ::Falcon::VMachine *vm );
FALCON_FUNC  strLower ( ::Falcon::VMachine *vm );
FALCON_FUNC  strCmpIgnoreCase ( ::Falcon::VMachine *vm );
FALCON_FUNC  strWildcardMatch ( ::Falcon::VMachine *vm );
FALCON_FUNC  strToMemBuf ( ::Falcon::VMachine *vm );
FALCON_FUNC  strFromMemBuf ( ::Falcon::VMachine *vm );


FALCON_FUNC  arrayIns ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayDel ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayDelAll ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayAdd ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayResize ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayBuffer ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayFind ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayScan ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayFilter ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayMap( ::Falcon::VMachine *vm );
FALCON_FUNC  arraySort( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayCopy( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayRemove( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayMerge( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayHead ( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayTail ( ::Falcon::VMachine *vm );

FALCON_FUNC  call( ::Falcon::VMachine *vm );
FALCON_FUNC  methodCall( ::Falcon::VMachine *vm );
FALCON_FUNC  marshalCB( ::Falcon::VMachine *vm );
FALCON_FUNC  marshalCBX( ::Falcon::VMachine *vm );
FALCON_FUNC  marshalCBR( ::Falcon::VMachine *vm );


FALCON_FUNC  bless( ::Falcon::VMachine *vm );
FALCON_FUNC  dictMerge( ::Falcon::VMachine *vm );
FALCON_FUNC  dictKeys( ::Falcon::VMachine *vm );
FALCON_FUNC  dictValues( ::Falcon::VMachine *vm );
FALCON_FUNC  dictInsert( ::Falcon::VMachine *vm );
FALCON_FUNC  dictGet( ::Falcon::VMachine *vm );
FALCON_FUNC  dictFind( ::Falcon::VMachine *vm );
FALCON_FUNC  dictBest( ::Falcon::VMachine *vm );
FALCON_FUNC  dictRemove( ::Falcon::VMachine *vm );
FALCON_FUNC  dictClear( ::Falcon::VMachine *vm );


FALCON_FUNC  fileType( ::Falcon::VMachine *vm );
FALCON_FUNC  fileNameSplit ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileNameMerge ( ::Falcon::VMachine *vm );
FALCON_FUNC  DirectoryOpen ( ::Falcon::VMachine *vm );
FALCON_FUNC  Directory_read ( ::Falcon::VMachine *vm );
FALCON_FUNC  Directory_close ( ::Falcon::VMachine *vm );
FALCON_FUNC  Directory_error ( ::Falcon::VMachine *vm );
FALCON_FUNC  dirChange ( ::Falcon::VMachine *vm );
FALCON_FUNC  dirCurrent ( ::Falcon::VMachine *vm );
FALCON_FUNC  dirMake ( ::Falcon::VMachine *vm );
FALCON_FUNC  dirRemove ( ::Falcon::VMachine *vm );
FALCON_FUNC  dirReadLink( ::Falcon::VMachine *vm );
FALCON_FUNC  dirMakeLink( ::Falcon::VMachine *vm );
FALCON_FUNC  fileMove ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileRemove ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileName ( ::Falcon::VMachine *vm );
FALCON_FUNC  filePath ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileChmod ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileChown ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileChgroup ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileCopy ( ::Falcon::VMachine *vm );

FALCON_FUNC  flc_random ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomChoice ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomPick ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomWalk ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomDice ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomSeed ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomGrab ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_fract ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_fint ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_round ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_floor ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_ceil ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_abs ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_log ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_exp ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_pow ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_sin ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_cos ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_tan ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_asin ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_acos ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_atan ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_atan2 ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_deg2rad ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_rad2deg ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_factorial ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_permutations ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_math_combinations ( ::Falcon::VMachine *vm );

FALCON_FUNC  serialize ( ::Falcon::VMachine *vm );
FALCON_FUNC  deserialize ( ::Falcon::VMachine *vm );

// Transcoder functions
FALCON_FUNC  transcodeTo ( ::Falcon::VMachine *vm );
FALCON_FUNC  transcodeFrom ( ::Falcon::VMachine *vm );
FALCON_FUNC  getSystemEncoding ( ::Falcon::VMachine *vm );

/* Timestamp class */
FALCON_FUNC  TimeStamp_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_currentTime ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_dayOfYear ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_dayOfWeek ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_toString ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_add ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_distance ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_isValid ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_isLeapYear ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_toLongFormat ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_fromLongFormat ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_compare ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_toRFC2822 ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_fromRFC2822 ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeStamp_changeZone ( ::Falcon::VMachine *vm );
FALCON_FUNC  CurrentTime ( ::Falcon::VMachine *vm );
FALCON_FUNC  ParseRFC2822 ( ::Falcon::VMachine *vm );

FALCON_FUNC  TimeZone_getDisplacement ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeZone_describe ( ::Falcon::VMachine *vm );
FALCON_FUNC  TimeZone_getLocal ( ::Falcon::VMachine *vm );

extern reflectionFuncDecl TimeStamp_timezone_rfrom;
extern reflectionFuncDecl TimeStamp_timezone_rto;

FALCON_FUNC  List_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_push ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_pop ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_pushFront ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_popFront ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_front ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_back ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_first ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_last ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_len( ::Falcon::VMachine *vm );
FALCON_FUNC  List_empty( ::Falcon::VMachine *vm );
FALCON_FUNC  List_erase ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_insert ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_clear ( ::Falcon::VMachine *vm );

FALCON_FUNC  CmdlineParser_parse( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_expectValue( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_terminate( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_usage( ::Falcon::VMachine *vm );

FALCON_FUNC FileReadStats ( ::Falcon::VMachine *vm ); // factory function
FALCON_FUNC FileStat_readStats ( ::Falcon::VMachine *vm );

reflectionFuncDecl FileStats_type_rfrom;
reflectionFuncDecl FileStats_mtime_rfrom;
reflectionFuncDecl FileStats_ctime_rfrom;
reflectionFuncDecl FileStats_atime_rfrom;

FALCON_FUNC  itemCopy( ::Falcon::VMachine *vm );
FALCON_FUNC  fal_include( ::Falcon::VMachine *vm );

/** Path class */
FALCON_FUNC Path_init ( ::Falcon::VMachine *vm );
reflectionFuncDecl Path_path_rfrom;
reflectionFuncDecl Path_filename_rfrom;
reflectionFuncDecl Path_file_rfrom;
reflectionFuncDecl Path_extension_rfrom;

reflectionFuncDecl Path_path_rto;
reflectionFuncDecl Path_filename_rto;
reflectionFuncDecl Path_file_rto;
reflectionFuncDecl Path_extension_rto;

/** URI class */
FALCON_FUNC  URI_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  URI_encode ( ::Falcon::VMachine *vm ); // static
FALCON_FUNC  URI_decode ( ::Falcon::VMachine *vm ); // static
FALCON_FUNC  URI_getFields ( ::Falcon::VMachine *vm );
FALCON_FUNC  URI_setFields ( ::Falcon::VMachine *vm );
reflectionFuncDecl URI_uri_rfrom;
reflectionFuncDecl URI_uri_rto;

/** Table class */
FALCON_FUNC  Table_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_setHeader ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_getHeader ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_getColData ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_order ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_len ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_front ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_back ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_first ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_last ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_get ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_columnPos ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_columnData ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_find ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_insert ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_remove ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_setColumn ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_insertColumn ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_removeColumn ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_choice ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_bidding ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_resetColumn ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_pageCount ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_setPage ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_curPage ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_insertPage ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_removePage ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_getPage ( ::Falcon::VMachine *vm );

/** Class used to manage the file stats.
   The FileStat reflection object provides three reflected timestamps
   in its object body.

   Although it is perfectly legal to:

   # Create an instance of TimeStamp falcon core object each time a
     reflected timestamp property is read; or

   # Derive the manager from FalconManager setting needCacheData() to return true, and then
     creating a TimeStamp reflected falcon object instance only the first time a TimeStamp
     property is asked for; or

   # Use the above method, but instead of reflecting the TimeStamp properties, setting them
     read only and creating the cached items at object creation.

   Those methods are sub-optimal, as they waste space. This object manager creates a class
   meant to store the fsdata and the cached TimeStamp instances that will be served on request.

   The cached TimeStamp items will hold a copy of Falcon::TimeStamp in them, so the FileStat
   instance can be destroyed and its TimeStamp will be destroyed with it. Notice that it
   would be possible to create object managers to use that instances directly, and keep alive
   their owner, but TimeStamp is too small to bother about this.
*/
class FileStatManager: public ObjectManager
{

public:

   class InnerData
   {
   public:
      FileStat m_fsdata;
      Item m_cache_atime;
      Item m_cache_mtime;
      Item m_cache_ctime;

      InnerData();
      InnerData( const InnerData &other );
      ~InnerData();
   };

   FileStatManager() {}
   virtual void *onInit( VMachine *vm );
   virtual void onGarbageMark( VMachine *vm, void *data );
   virtual void onDestroy( VMachine *vm, void *user_data );
   virtual void *onClone( VMachine *vm, void *user_data );
};


/** Special manager for URI reflection. */
class URIManager: public ObjectManager
{
public:

   // cache data is needed (not really needed, as we have non-read-only properties)
   virtual bool needCacheData() const { return true; }

   virtual void *onInit( VMachine *vm );
   virtual void onDestroy( VMachine *vm, void *user_data );
   virtual void *onClone( VMachine *vm, void *user_data );
   virtual bool onObjectReflectTo( CoreObject *reflector, void *user_data );
   virtual bool onObjectReflectFrom( CoreObject *reflector, void *user_data );
};

/** Special manager for URI reflection. */
class PathManager: public ObjectManager
{
public:

   // cache data is needed (not really needed, as we have non-read-only properties)
   virtual bool needCacheData() const { return true; }

   virtual void *onInit( VMachine *vm );
   virtual void onDestroy( VMachine *vm, void *user_data );
   virtual void *onClone( VMachine *vm, void *user_data );
   virtual bool onObjectReflectTo( CoreObject *reflector, void *user_data );
   virtual bool onObjectReflectFrom( CoreObject *reflector, void *user_data );
};


}}

#endif

/* end of core_module.h */
