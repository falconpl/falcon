/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_std.h
   $Id: falcon_rtl.cpp,v 1.15 2007/08/03 17:08:47 jonnymind Exp $

   Header for Falcon Realtime Library - C modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago  9 22:25:35 CEST 2004

   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/module.h>
#include <falcon/filestat.h>
#include <falcon/fstream.h>

#include "falcon_rtl_ext.h"
#include "version.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "falcon_rtl" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( RTL_VERSION_MAJOR, RTL_VERSION_MINOR, RTL_VERSION_REVISION );

   //=======================================================================
   // Message setting
   //=======================================================================
   // TODO: load proper messages...
   // ... and fallback to english:
   self->stringTable().build( Falcon::Ext::message_table );

   //=======================================================================
   // RTL basic functionality
   //=======================================================================

   self->addExtFunc( "print", Falcon::Ext::print );
   self->addExtFunc( "inspect", Falcon::Ext::inspect );
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
   self->addExtFunc( "marshallCB", Falcon::Ext::marshallCB );
   self->addExtFunc( "marshallCBX", Falcon::Ext::marshallCBX );
   self->addExtFunc( "marshallCBR", Falcon::Ext::marshallCBR );

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
   self->addClassMethod( stream_class, "close", Falcon::Ext::Stream_close );
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

   // inherits from stream.
   sstream_class->getClassDef()->addInheritance(  new Falcon::InheritDef( stream_class ) );

   // add methods
   self->addClassMethod( sstream_class, "getString", Falcon::Ext::StringStream_getString );
   self->addClassMethod( sstream_class, "closeToString", Falcon::Ext::StringStream_closeToString );

   //==============================================
   // The TimeStamp class -- declaration functional equivalent to
   // the one used for StringStream class (there in two steps, here in one).
   Falcon::Symbol *tstamp_class = self->addClass( "TimeStamp", Falcon::Ext::TimeStamp_init );

   // methods -- the first example is equivalent to the following.
   self->addClassMethod( tstamp_class, "currentTime",
      self->addExtFunc( "TimeStamp.currentTime", Falcon::Ext::TimeStamp_currentTime, false ) );

   self->addClassMethod( tstamp_class, "dayOfYear", Falcon::Ext::TimeStamp_dayOfYear );
   self->addClassMethod( tstamp_class, "dayOfWeek", Falcon::Ext::TimeStamp_dayOfWeek );
   self->addClassMethod( tstamp_class, "toString", Falcon::Ext::TimeStamp_toString );
   self->addClassMethod( tstamp_class, "add", Falcon::Ext::TimeStamp_add );
   self->addClassMethod( tstamp_class, "distance", Falcon::Ext::TimeStamp_distance );
   self->addClassMethod( tstamp_class, "isValid", Falcon::Ext::TimeStamp_isValid );
   self->addClassMethod( tstamp_class, "isLeapYear", Falcon::Ext::TimeStamp_isLeapYear );
   self->addClassMethod( tstamp_class, "toLongFormat", Falcon::Ext::TimeStamp_toLongFormat );
   self->addClassMethod( tstamp_class, "fromLongFormat", Falcon::Ext::TimeStamp_fromLongFormat );
   self->addClassMethod( tstamp_class, "compare", Falcon::Ext::TimeStamp_compare );

   // properties // setting to default 0
   self->addClassProperty( tstamp_class, "year" )->setInteger( 0 );
   self->addClassProperty( tstamp_class, "month" )->setInteger( 0 );
   self->addClassProperty( tstamp_class, "day" )->setInteger( 0 );
   self->addClassProperty( tstamp_class, "hour" )->setInteger( 0 );
   self->addClassProperty( tstamp_class, "minute" )->setInteger( 0 );
   self->addClassProperty( tstamp_class, "second" )->setInteger( 0 );
   self->addClassProperty( tstamp_class, "msec" )->setInteger( 0 );

   // A factory function that creates a timestamp already initialized to the current time:
   self->addExtFunc( "CurrentTime", Falcon::Ext::CurrentTime );

   //=======================================================================
   // Directory class
   //=======================================================================

   // factory function
   self->addExtFunc( "DirectoryOpen", Falcon::Ext::DirectoryOpen );

   Falcon::Symbol *dir_class = self->addClass( "Directory" );
   self->addClassMethod( dir_class, "read", Falcon::Ext::Directory_read );
   self->addClassMethod( dir_class, "close", Falcon::Ext::Directory_close );
   self->addClassMethod( dir_class, "error", Falcon::Ext::Directory_error );

   // Add the directory constants
   self->addConstant( "FILE_TYPE_NORMAL", (Falcon::int64) Falcon::FileStat::t_normal );
   self->addConstant( "FILE_TYPE_DIR", (Falcon::int64) Falcon::FileStat::t_dir );
   self->addConstant( "FILE_TYPE_PIPE", (Falcon::int64) Falcon::FileStat::t_pipe );
   self->addConstant( "FILE_TYPE_LINK", (Falcon::int64) Falcon::FileStat::t_link );
   self->addConstant( "FILE_TYPE_DEVICE", (Falcon::int64) Falcon::FileStat::t_device );
   self->addConstant( "FILE_TYPE_SOCKET", (Falcon::int64) Falcon::FileStat::t_socket );
   self->addConstant( "FILE_TYPE_UNKNOWN", (Falcon::int64) Falcon::FileStat::t_unknown );

   //=======================================================================
   // FileStat class
   //=======================================================================

   // factory function
   self->addExtFunc( "FileReadStats", Falcon::Ext::FileReadStats );

   // create the FileStat class (without constructor).
   Falcon::Symbol *fileStats_class = self->addClass( "FileStat" );

   // properties
   self->addClassProperty( fileStats_class, "type" );
   self->addClassProperty( fileStats_class, "size" );
   self->addClassProperty( fileStats_class, "owner" );
   self->addClassProperty( fileStats_class, "group" );
   self->addClassProperty( fileStats_class, "access" );
   self->addClassProperty( fileStats_class, "attribs" );
   self->addClassProperty( fileStats_class, "mtime" );
   self->addClassProperty( fileStats_class, "ctime" );
   self->addClassProperty( fileStats_class, "atime" );

   // methods
   self->addClassMethod( fileStats_class, "readStats",
            Falcon::Ext::FileStat_readStats );

   //=======================================================================
   // The list class
   //=======================================================================
   Falcon::Symbol *list_class = self->addClass( "List", Falcon::Ext::List_init );
   self->addClassMethod( list_class, "push", Falcon::Ext::List_push );
   self->addClassMethod( list_class, "pop", Falcon::Ext::List_pop );
   self->addClassMethod( list_class, "pushFront", Falcon::Ext::List_pushFront );
   self->addClassMethod( list_class, "popFront", Falcon::Ext::List_popFront );
   self->addClassMethod( list_class, "front", Falcon::Ext::List_front );
   self->addClassMethod( list_class, "back", Falcon::Ext::List_back );
   self->addClassMethod( list_class, "last", Falcon::Ext::List_last );
   self->addClassMethod( list_class, "first", Falcon::Ext::List_first );
   self->addClassMethod( list_class, "size", Falcon::Ext::List_size );
   self->addClassMethod( list_class, "empty", Falcon::Ext::List_empty );
   self->addClassMethod( list_class, "erase", Falcon::Ext::List_erase );
   self->addClassMethod( list_class, "insert", Falcon::Ext::List_insert );
   self->addClassMethod( list_class, "clear", Falcon::Ext::List_clear );

   //=======================================================================
   // The command line parser class
   //=======================================================================

   Falcon::Symbol *cmdparser_class = self->addClass( "CmdlineParser", true );
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
