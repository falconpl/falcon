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
#include <falcon/complex.h>

namespace Falcon {

class CoreObject;
class Error;

/* Utility function to generate RTL oriented errors */
Error *rtlError( int t, const String &desc );

namespace core {

FALCON_FUNC BOM_ptr( VMachine *vm );
FALCON_FUNC Integer_ptr( VMachine *vm );
FALCON_FUNC GarbagePointer_ptr( VMachine *vm );

FALCON_FUNC core_argv( VMachine *vm );
FALCON_FUNC core_argd( VMachine *vm );
FALCON_FUNC core_passvp( VMachine *vm );

// Methodic functions
FALCON_FUNC  mth_ToString ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_bound( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_len ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_typeId ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_compare ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_clone( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_serialize( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_isCallable ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_className ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_derivedFrom ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_baseClass ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_metaclass ( ::Falcon::VMachine *vm );

FALCON_FUNC  mth_getProperty( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_setProperty( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_properties( ::Falcon::VMachine *vm );
FALCON_FUNC  Dictionary_dop( ::Falcon::VMachine *vm );

//FALCON_FUNC  mth_hasProperty( ::Falcon::VMachine *vm );

FALCON_FUNC  Function_name ( ::Falcon::VMachine *vm );
FALCON_FUNC  Function_caller ( ::Falcon::VMachine *vm );
FALCON_FUNC  Function_trace ( ::Falcon::VMachine *vm );
FALCON_FUNC  Function_attributes( ::Falcon::VMachine *vm );

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
FALCON_FUNC  Iterator_compare( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_clone( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_erase( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_find( ::Falcon::VMachine *vm );
FALCON_FUNC  Iterator_insert( ::Falcon::VMachine *vm );

FALCON_FUNC  LateBinding_value( ::Falcon::VMachine *vm );
FALCON_FUNC  LateBinding_bound( ::Falcon::VMachine *vm );
FALCON_FUNC  LateBinding_bind( ::Falcon::VMachine *vm );
FALCON_FUNC  LateBinding_unbind( ::Falcon::VMachine *vm );

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
FALCON_FUNC  MessageError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  TableError_init ( ::Falcon::VMachine *vm );

FALCON_FUNC  val_int ( ::Falcon::VMachine *vm );
FALCON_FUNC  val_numeric ( ::Falcon::VMachine *vm );

FALCON_FUNC  attributes ( ::Falcon::VMachine *vm );
FALCON_FUNC  chr ( ::Falcon::VMachine *vm );
FALCON_FUNC  ord ( ::Falcon::VMachine *vm );


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

FALCON_FUNC  Format_parse ( ::Falcon::VMachine *vm );
FALCON_FUNC  Format_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Format_format ( ::Falcon::VMachine *vm );
FALCON_FUNC  Format_toString ( ::Falcon::VMachine *vm );

FALCON_FUNC  broadcast( ::Falcon::VMachine *vm );
FALCON_FUNC  subscribe( ::Falcon::VMachine *vm );
FALCON_FUNC  unsubscribe( ::Falcon::VMachine *vm );
FALCON_FUNC  getSlot( ::Falcon::VMachine *vm );
FALCON_FUNC  consume( ::Falcon::VMachine *vm );

FALCON_FUNC  assert( ::Falcon::VMachine *vm );
FALCON_FUNC  retract( ::Falcon::VMachine *vm );
FALCON_FUNC  getAssert( ::Falcon::VMachine *vm );

FALCON_FUNC  VMSlot_init( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_name( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_prepend( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_broadcast( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_subscribe( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_unsubscribe( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_assert( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_retract( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_getAssert( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_first( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_last( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_send( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_register( ::Falcon::VMachine *vm );
FALCON_FUNC  VMSlot_getEvent( ::Falcon::VMachine *vm );


FALCON_FUNC  core_exit ( ::Falcon::VMachine *vm );
FALCON_FUNC  PageDict( ::Falcon::VMachine *vm );
FALCON_FUNC Make_MemBuf( ::Falcon::VMachine *vm );
FALCON_FUNC Make_MemBufFromPtr( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_first( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_last( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_front( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_back( ::Falcon::VMachine *vm );

FALCON_FUNC MemoryBuffer_put( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_get( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_rewind( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_reset( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_flip( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_limit( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_mark( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_position( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_clear( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_fill( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_compact( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_remaining( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_wordSize( ::Falcon::VMachine *vm );
FALCON_FUNC MemoryBuffer_ptr( VMachine *vm );

FALCON_FUNC Method_source( ::Falcon::VMachine *vm );
FALCON_FUNC Method_base( ::Falcon::VMachine *vm );


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
FALCON_FUNC  core_upto ( ::Falcon::VMachine *vm );
FALCON_FUNC  core_downto ( ::Falcon::VMachine *vm );
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
FALCON_FUNC  core_brigade ( ::Falcon::VMachine *vm );

FALCON_FUNC  core_oob( ::Falcon::VMachine *vm );
FALCON_FUNC  core_deoob( ::Falcon::VMachine *vm );
FALCON_FUNC  core_isoob( ::Falcon::VMachine *vm );

FALCON_FUNC  core_lbind( ::Falcon::VMachine *vm );

reflectionFuncDecl GC_usedMem_rfrom;
reflectionFuncDecl GC_aliveMem_rfrom;
reflectionFuncDecl GC_items_rfrom;
reflectionFuncDecl GC_th_normal_rfrom;
reflectionFuncDecl GC_th_normal_rto;
reflectionFuncDecl GC_th_active_rfrom;
reflectionFuncDecl GC_th_active_rto;
CoreObject* GC_Factory( const CoreClass *cls, void *user_data, bool );

FALCON_FUNC  GC_adjust( ::Falcon::VMachine *vm );
FALCON_FUNC  GC_enable( ::Falcon::VMachine *vm );
FALCON_FUNC  GC_perform( ::Falcon::VMachine *vm );

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
FALCON_FUNC  vmSearchPath( ::Falcon::VMachine *vm );
FALCON_FUNC  vmModuleName( ::Falcon::VMachine *vm );
FALCON_FUNC  vmModulePath( ::Falcon::VMachine *vm );
FALCON_FUNC  vmRelativePath( ::Falcon::VMachine *vm );

FALCON_FUNC  print ( ::Falcon::VMachine *vm );
FALCON_FUNC  printl ( ::Falcon::VMachine *vm );
FALCON_FUNC  inspect ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_describe ( ::Falcon::VMachine *vm );
FALCON_FUNC  seconds ( ::Falcon::VMachine *vm );
FALCON_FUNC  input ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_getenv( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_setenv( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_unsetenv( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_getEnviron( ::Falcon::VMachine *vm );
FALCON_FUNC  InputStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  OutputStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  IOStream_creator ( ::Falcon::VMachine *vm );
FALCON_FUNC  systemErrorDescription ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_close ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_flush ( ::Falcon::VMachine *vm );
FALCON_FUNC  StdStream_close ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_read ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_grab ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readLine ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_grabLine ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_readText ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_grabText ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_write ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_writeText ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_setEncoding ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_clone ( ::Falcon::VMachine *vm );
FALCON_FUNC  readURI ( ::Falcon::VMachine *vm );
FALCON_FUNC  writeURI ( ::Falcon::VMachine *vm );


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
FALCON_FUNC  Stream_getBuffering ( ::Falcon::VMachine *vm );
FALCON_FUNC  Stream_setBuffering ( ::Falcon::VMachine *vm );

FALCON_FUNC  _stdIn ( ::Falcon::VMachine *vm );
FALCON_FUNC  _stdOut ( ::Falcon::VMachine *vm );
FALCON_FUNC  _stdErr ( ::Falcon::VMachine *vm );

FALCON_FUNC  stdInRaw ( ::Falcon::VMachine *vm );
FALCON_FUNC  stdOutRaw ( ::Falcon::VMachine *vm );
FALCON_FUNC  stdErrRaw ( ::Falcon::VMachine *vm );

FALCON_FUNC  StringStream_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  StringStream_getString ( ::Falcon::VMachine *vm );
FALCON_FUNC  StringStream_closeToString ( ::Falcon::VMachine *vm );


FALCON_FUNC  mth_strFront ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strBack ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strStartsWith ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strEndsWith ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strFill ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strSplit ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strSplitTrimmed ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strMerge ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strFind ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strBackFind ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strTrim ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strFrontTrim ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strBackTrim ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strReplace ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strReplicate ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strEsq ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strUnesq ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strEscape ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strUnescape ( ::Falcon::VMachine *vm );

FALCON_FUNC  strBuffer ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strUpper ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strLower ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strCmpIgnoreCase ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strWildcardMatch ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_strToMemBuf ( ::Falcon::VMachine *vm );
FALCON_FUNC  strFromMemBuf ( ::Falcon::VMachine *vm );
//FALCON_FUNC  String_first ( ::Falcon::VMachine *vm );
//FALCON_FUNC  String_last ( ::Falcon::VMachine *vm );
FALCON_FUNC  String_join ( ::Falcon::VMachine *vm );
FALCON_FUNC  String_ptr( VMachine *vm );
FALCON_FUNC  String_charSize( VMachine *vm );


FALCON_FUNC  mth_arrayIns ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayDel ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayDelAll ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayAdd ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayResize ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayFind ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayScan ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arraySort( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayRemove( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayMerge( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayHead ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayTail ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayFill ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayNM( ::Falcon::VMachine *vm );
FALCON_FUNC  arrayBuffer ( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_arrayCompact ( ::Falcon::VMachine *vm );

FALCON_FUNC  Array_comp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_mcomp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_mfcomp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_front ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_back ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_table ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_tabField ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_tabRow ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_first ( ::Falcon::VMachine *vm );
FALCON_FUNC  Array_last ( ::Falcon::VMachine *vm );

FALCON_FUNC  bless( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictMerge( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictKeys( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictValues( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictGet( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictSet( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictFind( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictBest( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictRemove( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictClear( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictFront( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictBack( ::Falcon::VMachine *vm );
FALCON_FUNC  mth_dictFill ( ::Falcon::VMachine *vm );
FALCON_FUNC  Dictionary_first( ::Falcon::VMachine *vm );
FALCON_FUNC  Dictionary_last( ::Falcon::VMachine *vm );
FALCON_FUNC  Dictionary_comp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Dictionary_mcomp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Dictionary_mfcomp ( ::Falcon::VMachine *vm );

FALCON_FUNC  fileType( ::Falcon::VMachine *vm );
FALCON_FUNC  fileNameSplit ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileNameMerge ( ::Falcon::VMachine *vm );
FALCON_FUNC  Directory_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Directory_read ( ::Falcon::VMachine *vm );
FALCON_FUNC  Directory_descend ( ::Falcon::VMachine *vm );
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
FALCON_FUNC  fileExt ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileUnit ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileChmod ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileChown ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileChgroup ( ::Falcon::VMachine *vm );
FALCON_FUNC  fileCopy ( ::Falcon::VMachine *vm );

FALCON_FUNC  flc_Random_init( ::Falcon::VMachine *vm );
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

FALCON_FUNC  Sequence_comp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_mcomp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_mfcomp ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_front ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_back ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_first ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_last ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_empty( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_clear ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_append ( ::Falcon::VMachine *vm );
FALCON_FUNC  Sequence_prepend ( ::Falcon::VMachine *vm );

FALCON_FUNC Continuation_init ( ::Falcon::VMachine *vm );
FALCON_FUNC Continuation_call ( ::Falcon::VMachine *vm );
FALCON_FUNC Continuation_reset ( ::Falcon::VMachine *vm );
FALCON_FUNC Continuation_complete ( ::Falcon::VMachine *vm );
FALCON_FUNC Continuation__suspend ( ::Falcon::VMachine *vm );

FALCON_FUNC  List_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_push ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_pop ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_pushFront ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_popFront ( ::Falcon::VMachine *vm );
FALCON_FUNC  List_len( ::Falcon::VMachine *vm );

FALCON_FUNC  Set_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Set_remove( ::Falcon::VMachine *vm );
FALCON_FUNC  Set_insert( ::Falcon::VMachine *vm );
FALCON_FUNC  Set_contains( ::Falcon::VMachine *vm );
FALCON_FUNC  Set_find( ::Falcon::VMachine *vm );
FALCON_FUNC  Set_len( ::Falcon::VMachine *vm );

FALCON_FUNC  CmdlineParser_parse( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_expectValue( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_terminate( ::Falcon::VMachine *vm );
FALCON_FUNC  CmdlineParser_usage( ::Falcon::VMachine *vm );

FALCON_FUNC FileStat_init ( ::Falcon::VMachine *vm );
FALCON_FUNC FileStat_read ( ::Falcon::VMachine *vm );

reflectionFuncDecl FileStats_type_rfrom;
reflectionFuncDecl FileStats_mtime_rfrom;
reflectionFuncDecl FileStats_ctime_rfrom;
reflectionFuncDecl FileStats_atime_rfrom;

FALCON_FUNC  fal_include( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_attributes( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_setState( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_getState( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_apply( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_retrieve( ::Falcon::VMachine *vm );
FALCON_FUNC  Method_attributes( ::Falcon::VMachine *vm );
FALCON_FUNC  Class_attributes( ::Falcon::VMachine *vm );

FALCON_FUNC  Object_comp( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_mcomp( ::Falcon::VMachine *vm );
FALCON_FUNC  Object_mfcomp( ::Falcon::VMachine *vm );

/** Path class */
FALCON_FUNC Path_init ( ::Falcon::VMachine *vm );
reflectionFuncDecl Path_path_rfrom;
reflectionFuncDecl Path_filename_rfrom;
reflectionFuncDecl Path_unit_rfrom;
reflectionFuncDecl Path_location_rfrom;
reflectionFuncDecl Path_fullloc_rfrom;
reflectionFuncDecl Path_file_rfrom;
reflectionFuncDecl Path_extension_rfrom;
reflectionFuncDecl Path_winpath_rfrom;
reflectionFuncDecl Path_winloc_rfrom;
reflectionFuncDecl Path_winfulloc_rfrom;

reflectionFuncDecl Path_path_rto;
reflectionFuncDecl Path_filename_rto;
reflectionFuncDecl Path_unit_rto;
reflectionFuncDecl Path_location_rto;
reflectionFuncDecl Path_fullloc_rto;
reflectionFuncDecl Path_file_rto;
reflectionFuncDecl Path_extension_rto;

CoreObject* PathObjectFactory( const CoreClass *cr, void *path, bool );


/** URI class */
FALCON_FUNC  URI_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  URI_encode ( ::Falcon::VMachine *vm ); // static
FALCON_FUNC  URI_decode ( ::Falcon::VMachine *vm ); // static
FALCON_FUNC  URI_toString ( ::Falcon::VMachine *vm );
FALCON_FUNC  URI_getFields ( ::Falcon::VMachine *vm );
FALCON_FUNC  URI_setFields ( ::Falcon::VMachine *vm );


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
FALCON_FUNC  Table_set ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_columnPos ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_columnData ( ::Falcon::VMachine *vm );

FALCON_FUNC  Table_find ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_insert ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_remove ( ::Falcon::VMachine *vm );
FALCON_FUNC  Table_append ( ::Falcon::VMachine *vm );

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

FALCON_FUNC  Tokenizer_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Tokenizer_parse ( ::Falcon::VMachine *vm );
FALCON_FUNC  Tokenizer_rewind ( ::Falcon::VMachine *vm );
FALCON_FUNC  Tokenizer_nextToken ( ::Falcon::VMachine *vm );
FALCON_FUNC  Tokenizer_next ( ::Falcon::VMachine *vm );
FALCON_FUNC  Tokenizer_token ( ::Falcon::VMachine *vm );
FALCON_FUNC  Tokenizer_hasCurrent ( ::Falcon::VMachine *vm );

#define TOKENIZER_OPT_GRROUPSEP 1
#define TOKENIZER_OPT_BINDSEP 2
#define TOKENIZER_OPT_TRIM 4
#define TOKENIZER_OPT_RSEP 8
#define TOKENIZER_OPT_WSISTOK 16


class UriObject: public CoreObject
{
public:
   UriObject( const CoreClass *gen ):
      CoreObject( gen )
   {}

   UriObject( const CoreClass *gen, const URI& uri ):
      CoreObject( gen ),
      m_uri( uri )
   {}

   UriObject( const UriObject &other );
   virtual ~UriObject();
   virtual UriObject *clone() const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &prop, Item &value ) const;

   const URI& uri() const { return m_uri; }
   URI& uri() { return m_uri; }

   static CoreObject* factory( const CoreClass *cr, void *uri, bool );
private:
   URI m_uri;
};


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
class FileStatObject: public ReflectObject
{

public:
   class InnerData
   {
   public:
      InnerData() {}
      InnerData( const InnerData &other );

      FileStat m_fsdata;
      Item m_cache_atime;
      Item m_cache_mtime;
      Item m_cache_ctime;
   };

   FileStatObject( const CoreClass* generator );
   FileStatObject( const FileStatObject &other );
   virtual ~FileStatObject();

   virtual void gcMark( uint32 mark );
   virtual FileStatObject* clone() const;

   InnerData* getInnerData() const { return (InnerData*) m_user_data; }
};

CoreObject* FileStatObjectFactory( const CoreClass *cls, void *user_data, bool bDeserializing );
CoreObject* PathObjectFactory( const CoreClass *me, void *uri, bool dyn );

/**********************************************
 * Carrier of complex number objects.
 */

class CoreComplex : public CoreObject
{
   Complex m_complex;

public:

   CoreComplex ( const CoreClass *cls ):
       CoreObject( cls )
   { }

   CoreComplex ( const Complex &origin, const CoreClass* cls ):
      CoreObject( cls ),
      m_complex( origin )
   {}

   CoreComplex ( const CoreComplex &other ):
      CoreObject( other ),
      m_complex( other.m_complex )
      {}

   virtual ~CoreComplex ();

   virtual void gcMark( uint32 ) { }
   virtual CoreObject *clone( void ) const { return new CoreComplex( *this ); }

   virtual bool hasProperty( const String &key ) const;
   virtual bool setProperty( const String &key, const Item &ret );
   virtual bool getProperty( const String &key, Item &ret ) const;

   const Complex &complex() const { return m_complex; }
   Complex &complex() { return m_complex; }
};

CoreObject *Complex_Factory( const CoreClass *cls, void *, bool );

FALCON_FUNC  Complex_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_toString( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_abs( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_conj( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_add__( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_sub__( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_mul__( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_div__( ::Falcon::VMachine *vm );
FALCON_FUNC  Complex_compare( ::Falcon::VMachine *vm );


}}

#endif

/* end of core_module.h */
