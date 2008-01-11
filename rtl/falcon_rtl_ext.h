/*
   FALCON - The Falcon Programming Language.
   FILE: falcon_rtl.h
   $Id: falcon_rtl_ext.h,v 1.13 2007/08/03 17:08:47 jonnymind Exp $

   Header for Falcon Realtime Library - C modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago  9 22:23:51 CEST 2004

   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef FALCON_STD_H
#define FALCON_STD_H

#include <falcon/module.h>
#include <falcon/timestamp.h>

namespace Falcon {

class CoreObject;
class Error;

/* Utility function to generate RTL oriented errors */
Error *rtlError( int t, const String &desc );

namespace Ext {

/**\file
   Falcon realtime library header file.

   Actually, falcon doesn't really need this file. The RTL library is both
   hard-linked in the stand alone falcon software (compiler, runner), or
   used as a plugin by embedding applications. Anyhow, embedding applications
   may wish to create hard-linked internal modules using already existing
   falcon RTL functions, so to include the source files or link against the shared
   object and provide just a part of the interface.

   It is possible to create a new module and add the prototypes that are listed
   here to reduce the functionality of falcon scripts, finetuining the ability of
   a script to change an environment. In example, an embedding application may
   wish to forbid the scripts the ability to write to disk, or to filter this
   ability through a self-provided criteria set. With the prototypes in this
   header it is possible to reduce, extend or warp RTL functionalities for an
   embedding application or a special implementation.

   Also, the RTL contains many functions that manages the Falcon items directly,
   an that may be useful also if used by an embedding application.
*/


/** Prints a list of items in the most simple format.

   Each item passed as a parameter is written on the standard output stream
   using its simplest representation form; every item is separated by the
   following with a space (ASCII 0x20). This function is quite simple, yet is
   quite powerful as it can be seen as a primitive to manage terminals in a
   very fast fashon.

   The output buffer is flushed at every write.

   So, this function can be quite useful both for fast raw output and
   debugging.

   \see printl
*/
FALCON_FUNC  print ( ::Falcon::VMachine *vm );

/** Prints a list of items and sends a newline.
   Work as print(), but sends an appropriate newline after all the items are written.

   \see falcon_ext_print
*/
FALCON_FUNC  printl ( ::Falcon::VMachine *vm );


/** Prints details on a signle object.
   Inspects an item and displays its internal status and data.

   \see falcon_ext_print
*/
FALCON_FUNC  inspect ( ::Falcon::VMachine *vm );

/** Returns the time of day in seconds and microseconds.
   Retruns a float number representing current seconds.
*/
FALCON_FUNC  seconds ( ::Falcon::VMachine *vm );

/** Reads a string from the console.
   VERY basic. Use only for testing.
*/
FALCON_FUNC  input ( ::Falcon::VMachine *vm );

/** Reads an environment variable

*/
FALCON_FUNC  falcon_getenv( ::Falcon::VMachine *vm );

/** Sets environment variables.

*/
FALCON_FUNC  falcon_setenv( ::Falcon::VMachine *vm );
/** Reads an environment variable

*/
FALCON_FUNC  falcon_unsetenv( ::Falcon::VMachine *vm );


/** Opens a file for reading.
   Format: InputStream( name )
*/
FALCON_FUNC  InputStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  OutputStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  IOStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  systemErrorDescription ( ::Falcon::VMachine *vm );

/** Closes a file. */
FALCON_FUNC  Stream_close ( ::Falcon::VMachine *vm );
FALCON_FUNC  StdStream_close ( ::Falcon::VMachine *vm );

/** Reads from a file. */
FALCON_FUNC  Stream_read ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readLine ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readText ( ::Falcon::VMachine *vm );

/** Writes to a file. */
FALCON_FUNC  Stream_write ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_writeText ( ::Falcon::VMachine *vm );

FALCON_FUNC  Stream_setEncoding ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_clone ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readItem ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_writeItem ( ::Falcon::VMachine *vm );

#define   CR_TO_CR 0
#define   CR_TO_CRLF 1
#define   SYSTEM_DETECT -1


/** Seeks a position in a file. */
FALCON_FUNC  Stream_seek ( ::Falcon::VMachine *vm );

/** Seeks a position in a file. */
FALCON_FUNC  Stream_seekEnd ( ::Falcon::VMachine *vm );

/** Seeks a position in a file. */
FALCON_FUNC  Stream_seekCur ( ::Falcon::VMachine *vm );

/** Return current position in a file. */
FALCON_FUNC  Stream_tell ( ::Falcon::VMachine *vm );

/** Truncate a file. */
FALCON_FUNC  Stream_truncate ( ::Falcon::VMachine *vm );

/** Return last hard-error on the file. */
FALCON_FUNC  Stream_lastError ( ::Falcon::VMachine *vm );

/** Return last quantity of sucessfully moved data. */
FALCON_FUNC  Stream_lastMoved ( ::Falcon::VMachine *vm );

/** Return a system dependent description of last error.. */
FALCON_FUNC  Stream_errorDescription ( ::Falcon::VMachine *vm );

/** Return true if at eof */
FALCON_FUNC  Stream_eof ( ::Falcon::VMachine *vm );

/** Return true if the file is open (ready to run) */
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

FALCON_FUNC  flc_random ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomChoice ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomPick ( ::Falcon::VMachine *vm );
FALCON_FUNC  flc_randomWalk ( ::Falcon::VMachine *vm );
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

FALCON_FUNC  serialize ( ::Falcon::VMachine *vm );
FALCON_FUNC  deserialize ( ::Falcon::VMachine *vm );

// Transcoder functions
FALCON_FUNC  transcodeTo ( ::Falcon::VMachine *vm );
FALCON_FUNC  transcodeFrom ( ::Falcon::VMachine *vm );
FALCON_FUNC  getSystemEncoding ( ::Falcon::VMachine *vm );

/** Timestamp class */
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
FALCON_FUNC  CurrentTime ( ::Falcon::VMachine *vm );


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

/** The command line parser */
FALCON_FUNC  CmdlineParser_parse( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_expectValue( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_terminate( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_usage( ::Falcon::VMachine *vm );

/** FileStat class */
FALCON_FUNC FileReadStats ( ::Falcon::VMachine *vm ); // factory function
FALCON_FUNC FileStat_readStats ( ::Falcon::VMachine *vm );

FALCON_FUNC  itemCopy( ::Falcon::VMachine *vm );

/** Message table **/
extern wchar_t *message_table[];



}}



#endif

/* end of falcon_rtl.h */
