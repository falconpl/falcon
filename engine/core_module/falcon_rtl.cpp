/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_std.h

   Header for Falcon Realtime Library - C modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago  9 22:25:35 CEST 2004


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/filestat.h>
#include <falcon/fstream.h>

#include "core_module.h"
#include "version.h"
#include <falcon/time_sys.h>
#include "core_messages.h"

using namespace Falcon;

static Ext::FileStatManager filestat_manager;
static Ext::URIManager uri_manager;
static Ext::PathManager path_manager;

/*#
   @brief Main Falcon module.

   The runtime library represents the set of standard commonly
   available functionalities that the Falcon users are usually
   expecting to be always loaded in the virtual machine. It includes
   important functions that manipulates types, as the array sorting
   function, or the dictionary traversal functions. Also, it provides basic
   I/O support, as file and directory manipulation.

   Some of the functions that are provided in the RTL, as printl, are a common
   hook for embedders. In the case of printl, which provides a common and comfortable
   simple output for scripts, embedders usually redirect it to their own functions to
   intercept script output request and redirect them. In some cases, embedders may
   may find useful to disable some functionalities, i.e. removing file/directory functions.

   RTL also provides many functions that duplicate operator functionalities.
   Array partitioning and string concatenations are some example. Those duplicate
   functions are meant both because they are usually more flexible than the pre-defined
   operators, and because, being Falcon a second order language, it is then possible to
   pass those functions as function parameters, or to save those functions in variables;
   of course, operators cannot be used that way.

   However, operators are faster than function calls, so, when possible, use the
   operators rather than the functions. Functions duplicating operators are mean to be
   used when operators cannot possibly be employed, or when the limited functionalities
   of the operators are not enough.

   As Falcon grows, some of this functions may be moved in more specific locations;
   in example, serialization, or system I/O functions, or mathematic functions, may
   be moved in other more specific modules. This depends on the decisions of the community
   for the best usage patterns.
*/

/*#
   @beginModule falcon_rtl
*/



FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   #define FALCON_DECLARE_MODULE self

   // setup DLL engine common data
   data.set();

   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "falcon_rtl" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( RTL_VERSION_MAJOR, RTL_VERSION_MINOR, RTL_VERSION_REVISION );


   //=======================================================================
   // Message setting
   //=======================================================================

   #include "core_messages.h"

   //=======================================================================
   // RTL basic functionality
   //=======================================================================

   self->addExtFunc( "print", Falcon::Ext::print );
   self->addExtFunc( "inspect", Falcon::Ext::inspect );
   self->addExtFunc( "inspectShort", Falcon::Ext::inspectShort );
   self->addExtFunc( "input", Falcon::Ext::input );
   self->addExtFunc( "printl", Falcon::Ext::printl );
   self->addExtFunc( "seconds", Falcon::Ext::seconds );

   //=======================================================================
   // RTL random api
   //=======================================================================

   self->addExtFunc( "random", Falcon::Ext::flc_random );
   self->addExtFunc( "randomChoice", Falcon::Ext::flc_randomChoice );
   self->addExtFunc( "randomPick", Falcon::Ext::flc_randomPick );
   self->addExtFunc( "randomWalk", Falcon::Ext::flc_randomWalk );
   self->addExtFunc( "randomGrab", Falcon::Ext::flc_randomGrab );
   self->addExtFunc( "randomSeed", Falcon::Ext::flc_randomSeed );
   self->addExtFunc( "randomDice", Falcon::Ext::flc_randomDice );

   //=======================================================================
   // RTL math
   //=======================================================================

   self->addExtFunc( "log", Falcon::Ext::flc_math_log );
   self->addExtFunc( "exp", Falcon::Ext::flc_math_exp );
   self->addExtFunc( "pow", Falcon::Ext::flc_math_pow );
   self->addExtFunc( "sin", Falcon::Ext::flc_math_sin );
   self->addExtFunc( "cos", Falcon::Ext::flc_math_cos );
   self->addExtFunc( "tan", Falcon::Ext::flc_math_tan );
   self->addExtFunc( "asin", Falcon::Ext::flc_math_asin );
   self->addExtFunc( "acos", Falcon::Ext::flc_math_acos );
   self->addExtFunc( "atan", Falcon::Ext::flc_math_atan );
   self->addExtFunc( "atan2", Falcon::Ext::flc_math_atan2 );
   self->addExtFunc( "rad2deg", Falcon::Ext::flc_math_rad2deg );
   self->addExtFunc( "deg2rad", Falcon::Ext::flc_math_deg2rad );
   self->addExtFunc( "fract", Falcon::Ext::flc_fract );
   self->addExtFunc( "fint", Falcon::Ext::flc_fint );
   self->addExtFunc( "round", Falcon::Ext::flc_round );
   self->addExtFunc( "floor", Falcon::Ext::flc_floor );
   self->addExtFunc( "ceil", Falcon::Ext::flc_ceil );
   self->addExtFunc( "abs", Falcon::Ext::flc_fract );

   //=======================================================================
   // RTL string api
   //=======================================================================
   self->addExtFunc( "strSplit", Falcon::Ext::strSplit );
   self->addExtFunc( "strSplitTrimmed", Falcon::Ext::strSplitTrimmed );
   self->addExtFunc( "strMerge", Falcon::Ext::strMerge );
   self->addExtFunc( "strFind", Falcon::Ext::strFind );
   self->addExtFunc( "strBackFind", Falcon::Ext::strBackFind );
   self->addExtFunc( "strFront", Falcon::Ext::strFront );
   self->addExtFunc( "strBack", Falcon::Ext::strBack );
   self->addExtFunc( "strTrim", Falcon::Ext::strTrim );
   self->addExtFunc( "strFrontTrim", Falcon::Ext::strFrontTrim );
   self->addExtFunc( "strAllTrim", Falcon::Ext::strAllTrim );
   self->addExtFunc( "strReplace", Falcon::Ext::strReplace );
   self->addExtFunc( "strReplicate", Falcon::Ext::strReplicate );
   self->addExtFunc( "strBuffer", Falcon::Ext::strBuffer );
   self->addExtFunc( "strUpper", Falcon::Ext::strUpper );
   self->addExtFunc( "strLower", Falcon::Ext::strLower );
   self->addExtFunc( "strCmpIgnoreCase", Falcon::Ext::strCmpIgnoreCase );
   self->addExtFunc( "strWildcardMatch", Falcon::Ext::strWildcardMatch );
   self->addExtFunc( "strToMemBuf", Falcon::Ext::strToMemBuf );
   self->addExtFunc( "strFromMemBuf", Falcon::Ext::strFromMemBuf );

   //=======================================================================
   // RTL array API
   //=======================================================================
   self->addExtFunc( "arrayIns", Falcon::Ext::arrayIns );
   self->addExtFunc( "arrayDel", Falcon::Ext::arrayDel );
   self->addExtFunc( "arrayDelAll", Falcon::Ext::arrayDelAll );
   self->addExtFunc( "arrayAdd", Falcon::Ext::arrayAdd );
   self->addExtFunc( "arrayResize", Falcon::Ext::arrayResize );
   self->addExtFunc( "arrayBuffer", Falcon::Ext::arrayBuffer );
   self->addExtFunc( "arrayFind", Falcon::Ext::arrayFind );
   self->addExtFunc( "arrayScan", Falcon::Ext::arrayScan );
   self->addExtFunc( "arrayFilter", Falcon::Ext::arrayFilter );
   self->addExtFunc( "arrayMap", Falcon::Ext::arrayMap );
   self->addExtFunc( "arraySort", Falcon::Ext::arraySort );
   self->addExtFunc( "arrayCopy", Falcon::Ext::arrayCopy );
   self->addExtFunc( "arrayRemove", Falcon::Ext::arrayRemove );
   self->addExtFunc( "arrayMerge", Falcon::Ext::arrayMerge );
   self->addExtFunc( "arrayHead", Falcon::Ext::arrayHead );
   self->addExtFunc( "arrayTail", Falcon::Ext::arrayTail );

   //=======================================================================
   // Indirect call
   //=======================================================================
   self->addExtFunc( "call", Falcon::Ext::call );
   self->addExtFunc( "methodCall", Falcon::Ext::methodCall );
   self->addExtFunc( "marshalCB", Falcon::Ext::marshalCB );
   self->addExtFunc( "marshalCBX", Falcon::Ext::marshalCBX );
   self->addExtFunc( "marshalCBR", Falcon::Ext::marshalCBR );

   //=======================================================================
   // RTL dictionary
   //=======================================================================
   self->addExtFunc( "dictMerge", Falcon::Ext::dictMerge );
   self->addExtFunc( "dictKeys", Falcon::Ext::dictKeys );
   self->addExtFunc( "dictValues", Falcon::Ext::dictValues );
   self->addExtFunc( "dictInsert", Falcon::Ext::dictInsert );
   self->addExtFunc( "dictGet", Falcon::Ext::dictGet );
   self->addExtFunc( "dictFind", Falcon::Ext::dictFind );
   self->addExtFunc( "dictBest", Falcon::Ext::dictBest );
   self->addExtFunc( "dictRemove", Falcon::Ext::dictRemove );
   self->addExtFunc( "dictClear", Falcon::Ext::dictClear );

   self->addExtFunc( "fileType", Falcon::Ext::fileType );
   self->addExtFunc( "fileNameMerge", Falcon::Ext::fileNameMerge );
   self->addExtFunc( "fileNameSplit", Falcon::Ext::fileNameSplit );
   self->addExtFunc( "fileName", Falcon::Ext::fileName );
   self->addExtFunc( "filePath", Falcon::Ext::filePath );
   self->addExtFunc( "fileMove", Falcon::Ext::fileMove );
   self->addExtFunc( "fileRemove", Falcon::Ext::fileRemove );
   self->addExtFunc( "fileChown", Falcon::Ext::fileChown );
   self->addExtFunc( "fileChmod", Falcon::Ext::fileChmod );
   self->addExtFunc( "fileChgroup", Falcon::Ext::fileChgroup );
   self->addExtFunc( "fileCopy", Falcon::Ext::fileCopy );

   self->addExtFunc( "dirMake", Falcon::Ext::dirMake );
   self->addExtFunc( "dirChange", Falcon::Ext::dirChange );
   self->addExtFunc( "dirCurrent", Falcon::Ext::dirCurrent );
   self->addExtFunc( "dirRemove", Falcon::Ext::dirRemove );
   self->addExtFunc( "dirReadLink", Falcon::Ext::dirReadLink );
   self->addExtFunc( "dirMakeLink", Falcon::Ext::dirMakeLink );

   self->addExtFunc( "serialize", Falcon::Ext::serialize );
   self->addExtFunc( "deserialize", Falcon::Ext::deserialize );

   self->addExtFunc( "itemCopy", Falcon::Ext::itemCopy );

   //==============================================
   // Transcoding functions

   self->addExtFunc( "transcodeTo", Falcon::Ext::transcodeTo );
   self->addExtFunc( "transcodeTo", Falcon::Ext::transcodeFrom );
   self->addExtFunc( "getSystemEncoding", Falcon::Ext::getSystemEncoding );

   //==============================================
   // Environment variable functions

   self->addExtFunc( "getenv", Falcon::Ext::falcon_getenv );
   self->addExtFunc( "setenv", Falcon::Ext::falcon_setenv );
   self->addExtFunc( "unsetenv", Falcon::Ext::falcon_unsetenv );
   //=======================================================================
   // RTL CLASSES
   //=======================================================================

   //==============================================
   // Stream class

   // Factory functions
   self->addExtFunc( "InputStream", Falcon::Ext::InputStream_creator );
   self->addExtFunc( "OutputStream", Falcon::Ext::OutputStream_creator );
   self->addExtFunc( "IOStream", Falcon::Ext::IOStream_creator );

   // create the stream class (without constructor).
   Falcon::Symbol *stream_class = self->addClass( "Stream" );
   stream_class->setWKS(true);
   stream_class->getClassDef()->setObjectManager( &core_falcon_data_manager );
   self->addClassMethod( stream_class, "close", Falcon::Ext::Stream_close );
   self->addClassMethod( stream_class, "flush", Falcon::Ext::Stream_flush );
   self->addClassMethod( stream_class, "read", Falcon::Ext::Stream_read );
   self->addClassMethod( stream_class, "readLine", Falcon::Ext::Stream_readLine );
   self->addClassMethod( stream_class, "write", Falcon::Ext::Stream_write );
   self->addClassMethod( stream_class, "seek", Falcon::Ext::Stream_seek );
   self->addClassMethod( stream_class, "seekEnd", Falcon::Ext::Stream_seekEnd );
   self->addClassMethod( stream_class, "seekCur", Falcon::Ext::Stream_seekCur );
   self->addClassMethod( stream_class, "tell", Falcon::Ext::Stream_tell );
   self->addClassMethod( stream_class, "truncate", Falcon::Ext::Stream_truncate );
   self->addClassMethod( stream_class, "lastMoved", Falcon::Ext::Stream_lastMoved );
   self->addClassMethod( stream_class, "lastError", Falcon::Ext::Stream_lastError );
   self->addClassMethod( stream_class, "errorDescription", Falcon::Ext::Stream_errorDescription );
   self->addClassMethod( stream_class, "eof", Falcon::Ext::Stream_eof );
   self->addClassMethod( stream_class, "isOpen", Falcon::Ext::Stream_errorDescription );
   self->addClassMethod( stream_class, "readAvailable", Falcon::Ext::Stream_readAvailable );
   self->addClassMethod( stream_class, "writeAvailable", Falcon::Ext::Stream_writeAvailable );
   self->addClassMethod( stream_class, "readText", Falcon::Ext::Stream_readText );
   self->addClassMethod( stream_class, "writeText", Falcon::Ext::Stream_writeText );
   self->addClassMethod( stream_class, "setEncoding", Falcon::Ext::Stream_setEncoding );
   self->addClassMethod( stream_class, "clone", Falcon::Ext::Stream_clone );
   self->addClassMethod( stream_class, "readItem", Falcon::Ext::Stream_readItem );
   self->addClassMethod( stream_class, "writeItem", Falcon::Ext::Stream_writeItem );

   // Specialization of the stream class to manage the closing of process bound streams.
   Falcon::Symbol *stdstream_class = self->addClass( "StdStream" );
   stdstream_class->setWKS(true);
   self->addClassMethod( stdstream_class, "close", Falcon::Ext::StdStream_close );
   self->addClassProperty( stdstream_class, "_stdStreamType" );
   stdstream_class->getClassDef()->addInheritance(  new Falcon::InheritDef( stream_class ) );


   self->addConstant( "FILE_EXCLUSIVE", (Falcon::int64) Falcon::GenericStream::e_smExclusive );
   self->addConstant( "FILE_SHARE_READ", (Falcon::int64) Falcon::GenericStream::e_smShareRead );
   self->addConstant( "FILE_SHARE", (Falcon::int64) Falcon::GenericStream::e_smShareFull );

   self->addConstant( "CR_TO_CR", (Falcon::int64) CR_TO_CR );
   self->addConstant( "CR_TO_CRLF", (Falcon::int64) CR_TO_CRLF );
   self->addConstant( "SYSTEM_DETECT", (Falcon::int64) SYSTEM_DETECT );

   //==============================================
   // StringStream class

   Falcon::Symbol *sstream_ctor = self->addExtFunc( "StringStream._init",
               Falcon::Ext::StringStream_init, false );
   Falcon::Symbol *sstream_class = self->addClass( "StringStream", sstream_ctor );
   sstream_class->setWKS(true);

   // inherits from stream.
   sstream_class->getClassDef()->addInheritance(  new Falcon::InheritDef( stream_class ) );

   // add methods
   self->addClassMethod( sstream_class, "getString", Falcon::Ext::StringStream_getString );
   self->addClassMethod( sstream_class, "closeToString", Falcon::Ext::StringStream_closeToString );

   //==============================================
   // The TimeStamp class -- declaration functional equivalent to
   // the one used for StringStream class (there in two steps, here in one).
   Falcon::Symbol *tstamp_class = self->addClass( "TimeStamp", Falcon::Ext::TimeStamp_init );
   tstamp_class->setWKS( true );
   tstamp_class->getClassDef()->setObjectManager( &core_falcon_data_manager );

   // methods -- the first example is equivalent to the following.
   self->addClassMethod( tstamp_class, "currentTime",
      self->addExtFunc( "TimeStamp.currentTime", Falcon::Ext::TimeStamp_currentTime, false ) ).setReadOnly(true);

   self->addClassMethod( tstamp_class, "dayOfYear", Falcon::Ext::TimeStamp_dayOfYear ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "dayOfWeek", Falcon::Ext::TimeStamp_dayOfWeek ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "toString", Falcon::Ext::TimeStamp_toString ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "add", Falcon::Ext::TimeStamp_add ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "distance", Falcon::Ext::TimeStamp_distance ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "isValid", Falcon::Ext::TimeStamp_isValid ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "isLeapYear", Falcon::Ext::TimeStamp_isLeapYear ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "toLongFormat", Falcon::Ext::TimeStamp_toLongFormat ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "fromLongFormat", Falcon::Ext::TimeStamp_fromLongFormat ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "compare", Falcon::Ext::TimeStamp_compare ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "fromRFC2822", Falcon::Ext::TimeStamp_fromRFC2822 ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "toRFC2822", Falcon::Ext::TimeStamp_toRFC2822 ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "changeZone", Falcon::Ext::TimeStamp_changeZone ).setReadOnly(true);

   // properties
   TimeStamp ts_dummy;
   self->addClassProperty( tstamp_class, "year" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_year );
   self->addClassProperty( tstamp_class, "month" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_month );
   self->addClassProperty( tstamp_class, "day" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_day );
   self->addClassProperty( tstamp_class, "hour" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_hour );
   self->addClassProperty( tstamp_class, "minute" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_minute );
   self->addClassProperty( tstamp_class, "second" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_second );
   self->addClassProperty( tstamp_class, "msec" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_msec );
   self->addClassProperty( tstamp_class, "timezone" ).
      setReflectFunc( Falcon::Ext::TimeStamp_timezone_rfrom, Falcon::Ext::TimeStamp_timezone_rto );

   Falcon::Symbol *c_timezone = self->addClass( "TimeZone" );
   self->addClassMethod( c_timezone, "getDisplacement", Falcon::Ext::TimeZone_getDisplacement );
   self->addClassMethod( c_timezone, "describe", Falcon::Ext::TimeZone_describe );
   self->addClassMethod( c_timezone, "getLocal", Falcon::Ext::TimeZone_getLocal );
   self->addClassProperty( c_timezone, "local" ).setInteger( Falcon::tz_local );
   self->addClassProperty( c_timezone, "UTC" ).setInteger( Falcon::tz_UTC );
   self->addClassProperty( c_timezone, "GMT" ).setInteger( Falcon::tz_UTC );
   self->addClassProperty( c_timezone, "E1" ).setInteger( Falcon::tz_UTC_E_1 );
   self->addClassProperty( c_timezone, "E2" ).setInteger( Falcon::tz_UTC_E_2 );
   self->addClassProperty( c_timezone, "E3" ).setInteger( Falcon::tz_UTC_E_3 );
   self->addClassProperty( c_timezone, "E4" ).setInteger( Falcon::tz_UTC_E_4 );
   self->addClassProperty( c_timezone, "E5" ).setInteger( Falcon::tz_UTC_E_5 );
   self->addClassProperty( c_timezone, "E6" ).setInteger( Falcon::tz_UTC_E_6 );
   self->addClassProperty( c_timezone, "E7" ).setInteger( Falcon::tz_UTC_E_7 );
   self->addClassProperty( c_timezone, "E8" ).setInteger( Falcon::tz_UTC_E_8 );
   self->addClassProperty( c_timezone, "E9" ).setInteger( Falcon::tz_UTC_E_9 );
   self->addClassProperty( c_timezone, "E10" ).setInteger( Falcon::tz_UTC_E_10 );
   self->addClassProperty( c_timezone, "E11" ).setInteger( Falcon::tz_UTC_E_11 );
   self->addClassProperty( c_timezone, "E12" ).setInteger( Falcon::tz_UTC_E_12 );

   self->addClassProperty( c_timezone, "W1" ).setInteger( Falcon::tz_UTC_W_1 );
   self->addClassProperty( c_timezone, "W2" ).setInteger( Falcon::tz_UTC_W_2 );
   self->addClassProperty( c_timezone, "W3" ).setInteger( Falcon::tz_UTC_W_3 );
   self->addClassProperty( c_timezone, "W4" ).setInteger( Falcon::tz_UTC_W_4 );
   self->addClassProperty( c_timezone, "EDT" ).setInteger( Falcon::tz_UTC_W_4 );
   self->addClassProperty( c_timezone, "W5" ).setInteger( Falcon::tz_UTC_W_5 );
   self->addClassProperty( c_timezone, "EST" ).setInteger( Falcon::tz_UTC_W_5 );
   self->addClassProperty( c_timezone, "CDT" ).setInteger( Falcon::tz_UTC_W_5 );
   self->addClassProperty( c_timezone, "W6" ).setInteger( Falcon::tz_UTC_W_6 );
   self->addClassProperty( c_timezone, "CST" ).setInteger( Falcon::tz_UTC_W_6 );
   self->addClassProperty( c_timezone, "MDT" ).setInteger( Falcon::tz_UTC_W_6 );
   self->addClassProperty( c_timezone, "W7" ).setInteger( Falcon::tz_UTC_W_7 );
   self->addClassProperty( c_timezone, "MST" ).setInteger( Falcon::tz_UTC_W_7 );
   self->addClassProperty( c_timezone, "PDT" ).setInteger( Falcon::tz_UTC_W_7 );
   self->addClassProperty( c_timezone, "W8" ).setInteger( Falcon::tz_UTC_W_8 );
   self->addClassProperty( c_timezone, "PST" ).setInteger( Falcon::tz_UTC_W_8 );
   self->addClassProperty( c_timezone, "W9" ).setInteger( Falcon::tz_UTC_W_9 );
   self->addClassProperty( c_timezone, "W10" ).setInteger( Falcon::tz_UTC_W_10 );
   self->addClassProperty( c_timezone, "W11" ).setInteger( Falcon::tz_UTC_W_11 );
   self->addClassProperty( c_timezone, "W12" ).setInteger( Falcon::tz_UTC_W_12 );

   self->addClassProperty( c_timezone, "NFT" ).setInteger( Falcon::tz_NFT );
   self->addClassProperty( c_timezone, "ACDT" ).setInteger( Falcon::tz_ACDT );
   self->addClassProperty( c_timezone, "ACST" ).setInteger( Falcon::tz_ACST );
   self->addClassProperty( c_timezone, "HAT" ).setInteger( Falcon::tz_HAT );
   self->addClassProperty( c_timezone, "NST" ).setInteger( Falcon::tz_NST );

   self->addClassProperty( c_timezone, "NONE" ).setInteger( Falcon::tz_NST );

   // A factory function that creates a timestamp already initialized to the current time:
   self->addExtFunc( "CurrentTime", Falcon::Ext::CurrentTime );
   self->addExtFunc( "ParseRFC2822", Falcon::Ext::ParseRFC2822 );

   //=======================================================================
   // Directory class
   //=======================================================================

   // factory function
   self->addExtFunc( "DirectoryOpen", Falcon::Ext::DirectoryOpen );

   Falcon::Symbol *dir_class = self->addClass( "Directory" );
   dir_class->setWKS(true);
   dir_class->getClassDef()->setObjectManager( &core_falcon_data_manager );
   self->addClassMethod( dir_class, "read", Falcon::Ext::Directory_read );
   self->addClassMethod( dir_class, "close", Falcon::Ext::Directory_close );
   self->addClassMethod( dir_class, "error", Falcon::Ext::Directory_error );

   // Add the directory constants

   //=======================================================================
   // FileStat class
   //=======================================================================

   // factory function
   self->addExtFunc( "FileReadStats", Falcon::Ext::FileReadStats );

   // create the FileStat class (without constructor).
   Falcon::Symbol *fileStats_class = self->addClass( "FileStat" );
   fileStats_class->setWKS( true );
   fileStats_class->getClassDef()->setObjectManager( &filestat_manager );

   // properties
   Ext::FileStatManager::InnerData id;
   self->addClassProperty( fileStats_class, "ftype" ).
      setReflectFunc( Falcon::Ext::FileStats_type_rfrom ); // read only, we have no set.
   self->addClassProperty( fileStats_class, "size" ).setReflective( e_reflectLL, &id, &id.m_fsdata.m_size );
   self->addClassProperty( fileStats_class, "owner" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_owner );
   self->addClassProperty( fileStats_class, "group" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_group );
   self->addClassProperty( fileStats_class, "access" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_access );
   self->addClassProperty( fileStats_class, "attribs" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_attribs );
   self->addClassProperty( fileStats_class, "mtime" ).
      setReflectFunc( Falcon::Ext::FileStats_mtime_rfrom );
   self->addClassProperty( fileStats_class, "ctime" ).
      setReflectFunc( Falcon::Ext::FileStats_ctime_rfrom );
   self->addClassProperty( fileStats_class, "atime" ).
      setReflectFunc( Falcon::Ext::FileStats_atime_rfrom );

   self->addClassProperty( fileStats_class, "NORMAL" ).setInteger( (Falcon::int64) Falcon::FileStat::t_normal ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "DIR" ).setInteger( (Falcon::int64) Falcon::FileStat::t_dir ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "PIPE" ).setInteger( (Falcon::int64) Falcon::FileStat::t_pipe ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "LINK" ).setInteger( (Falcon::int64) Falcon::FileStat::t_link ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "DEVICE" ).setInteger( (Falcon::int64) Falcon::FileStat::t_device ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "SOCKET" ).setInteger( (Falcon::int64) Falcon::FileStat::t_socket ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "UNKNOWN" ).setInteger( (Falcon::int64) Falcon::FileStat::t_unknown ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "NOTFOUND" ).setInteger( (Falcon::int64) Falcon::FileStat::t_notFound ).
      setReadOnly( true );

   // methods - set read only to have full reflection
   self->addClassMethod( fileStats_class, "readStats",
            Falcon::Ext::FileStat_readStats ).setReadOnly(true);

   //=======================================================================
   // The list class
   //=======================================================================
   Falcon::Symbol *list_class = self->addClass( "List", Falcon::Ext::List_init );
   list_class->setWKS(true);
   list_class->getClassDef()->setObjectManager( &core_falcon_data_manager );
   self->addClassMethod( list_class, "push", Falcon::Ext::List_push );
   self->addClassMethod( list_class, "pop", Falcon::Ext::List_pop );
   self->addClassMethod( list_class, "pushFront", Falcon::Ext::List_pushFront );
   self->addClassMethod( list_class, "popFront", Falcon::Ext::List_popFront );
   self->addClassMethod( list_class, "front", Falcon::Ext::List_front );
   self->addClassMethod( list_class, "back", Falcon::Ext::List_back );
   self->addClassMethod( list_class, "last", Falcon::Ext::List_last );
   self->addClassMethod( list_class, "first", Falcon::Ext::List_first );
   self->addClassMethod( list_class, "len", Falcon::Ext::List_len );
   self->addClassMethod( list_class, "empty", Falcon::Ext::List_empty );
   self->addClassMethod( list_class, "erase", Falcon::Ext::List_erase );
   self->addClassMethod( list_class, "insert", Falcon::Ext::List_insert );
   self->addClassMethod( list_class, "clear", Falcon::Ext::List_clear );

   //=======================================================================
   // The path class
   //=======================================================================
   Falcon::Symbol *path_class = self->addClass( "Path", Falcon::Ext::Path_init );
   path_class->getClassDef()->setObjectManager( &path_manager );
   path_class->setWKS(true);

   self->addClassProperty( path_class, "path" ).
         setReflectFunc( Falcon::Ext::Path_path_rfrom, Falcon::Ext::Path_path_rto );
   self->addClassProperty( path_class, "unit" );
   self->addClassProperty( path_class, "location" );
   self->addClassProperty( path_class, "file" ).
         setReflectFunc( Falcon::Ext::Path_file_rfrom, Falcon::Ext::Path_file_rto );
   self->addClassProperty( path_class, "extension" ).
         setReflectFunc( Falcon::Ext::Path_extension_rfrom, Falcon::Ext::Path_extension_rto );
   self->addClassProperty( path_class, "filename" ).
         setReflectFunc( Falcon::Ext::Path_filename_rfrom, Falcon::Ext::Path_filename_rto );

   //=======================================================================
   // The path class
   //=======================================================================
   Falcon::Symbol *uri_class = self->addClass( "URI", Falcon::Ext::URI_init );
   uri_class->getClassDef()->setObjectManager( &uri_manager );
   uri_class->setWKS(true);

   self->addClassProperty( uri_class, "scheme" );
   self->addClassProperty( uri_class, "userInfo" );
   self->addClassProperty( uri_class, "host" );
   self->addClassProperty( uri_class, "port" );
   self->addClassProperty( uri_class, "path" );
   self->addClassProperty( uri_class, "query" );
   self->addClassProperty( uri_class, "fragment" );
   self->addClassProperty( uri_class, "uri" ).
         setReflectFunc( Falcon::Ext::URI_uri_rfrom, Falcon::Ext::URI_uri_rto );
   self->addClassMethod( uri_class, "encode", Falcon::Ext::URI_encode );
   self->addClassMethod( uri_class, "decode", Falcon::Ext::URI_decode );
   self->addClassMethod( uri_class, "getFields", Falcon::Ext::URI_getFields );
   self->addClassMethod( uri_class, "setFields", Falcon::Ext::URI_setFields );

   //=======================================================================
   // The command line parser class
   //=======================================================================

   Falcon::Symbol *cmdparser_class = self->addClass( "CmdlineParser", true );
   cmdparser_class->getClassDef()->setObjectManager( &core_falcon_data_manager );
   self->addClassMethod( cmdparser_class, "parse", Falcon::Ext::CmdlineParser_parse );
   self->addClassMethod( cmdparser_class, "expectValue", Falcon::Ext::CmdlineParser_expectValue );
   self->addClassMethod( cmdparser_class, "terminate", Falcon::Ext::CmdlineParser_terminate );
   // private property internally used to communicate between the child classes and
   // the base parse.
   self->addClassProperty( cmdparser_class, "_request" );
   // Properties that will hold callbacks
   self->addClassProperty( cmdparser_class, "onOption" );
   self->addClassProperty( cmdparser_class, "onFree" );
   self->addClassProperty( cmdparser_class, "onValue" );
   self->addClassProperty( cmdparser_class, "onSwitchOff" );
   self->addClassProperty( cmdparser_class, "passMinusMinus" );
   self->addClassProperty( cmdparser_class, "lastParsed" );
   self->addClassMethod( cmdparser_class, "usage", Falcon::Ext::CmdlineParser_usage );


   //=======================================================================
   // SYSTEM API
   //=======================================================================
   self->addExtFunc( "stdIn", Falcon::Ext::_stdIn );
   self->addExtFunc( "stdOut", Falcon::Ext::_stdOut );
   self->addExtFunc( "stdErr", Falcon::Ext::_stdErr );
   self->addExtFunc( "stdInRaw", Falcon::Ext::stdInRaw );
   self->addExtFunc( "stdOutRaw", Falcon::Ext::stdOutRaw );
   self->addExtFunc( "stdErrRaw", Falcon::Ext::stdErrRaw );
   self->addExtFunc( "systemErrorDescription", Falcon::Ext::systemErrorDescription );

   return self;
}

/* end of falcon_rtl.cpp */
