/*
   FALCON - The Falcon Programming Language.
   FILE: error.h
   $Id: error.h,v 1.17 2007/08/18 17:59:33 jonnymind Exp $

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom feb 18 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Error class definition file.
   (this file contains also the TraceStep class).
*/

#ifndef flc_error_H
#define flc_error_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/error.h>
#include <falcon/item.h>
#include <falcon/genericlist.h>
#include <falcon/string.h>
#include <falcon/userdata.h>

namespace Falcon {

const int e_none =                 0;
const int e_syntax =               1;
const int e_unpack_size =          2;
const int e_break_out =            3;
const int e_continue_out =         4;
const int e_div_by_zero =          6;
const int e_mod_by_zero =          7;
const int e_invalid_op =           8;
const int e_assign_const =         9;
const int e_assign_sym =           10;
const int e_repeated =             11;
const int e_global_notin_func =    12;
const int e_already_def =          13;
const int e_non_callable =         14;
const int e_invalid_cmp =          15;
const int e_export_undef =         16;
const int e_export_all =           17;
const int e_misplaced_stat =       18;
const int e_enter_outside =        19;
const int e_leave_outside =        20;
const int e_static_notin_func =    21;
const int e_self_outclass =        22;
const int e_sender_outclass =      23;
const int e_undef_sym  =           24;
const int e_invops =               25;
const int e_no_local =             26;
const int e_too_entry =            27;
const int e_end_no_loc =           28;
const int e_import_out =           29;
const int e_no_import =            30;
const int e_too_locals =           31;
const int e_too_params =           32;
const int e_switch_again =         33;
const int e_switch_case =          34;
const int e_switch_end =           35;
const int e_inv_setstate =         36;
const int e_prop_no_class =        37;
const int e_prop_pinit =           38;
const int e_prop_adef =            39;
const int e_too_props =            40;
const int e_from_adef =            41;
const int e_too_froms =            42;
const int e_invopcode =            43;
const int e_invop =                44;
const int e_stackuf =              45;
const int e_stackof =              46;
const int e_arracc =               47;
const int e_nostartsym =           48;
const int e_uncaught =             49;
const int e_binload =              50;
const int e_binstartup =           51;
const int e_bininit =              52;
const int e_modver =               53;
const int e_modformat =            54;
const int e_modio =                55;
const int e_unclosed_cs =          56;
const int e_runaway_eof =          57;
const int e_cmp_unprep =           58;
const int e_undef_label =          59;
const int e_prop_acc =             60;
const int e_deadlock =             61;
const int e_prov_name =            62;
const int e_dup_case =             63;
const int e_init_given =           64;
const int e_static_const =         65;
const int e_str_noid =             66;
const int e_inv_inherit =          67;
const int e_nonsym_ref =           68;
const int e_state_adef =           69;
const int e_invalid_sjmp =         100;
const int e_no_attrib =            101;
const int e_no_cls_inst =          102;
const int e_pass_outside =         103;
const int e_switch_clash =         104;
const int e_switch_default =       105;
const int e_for_user_error =       106;
const int e_global_again =         107;
const int e_service_adef =         108;
const int e_service_undef =        109;
const int e_uncloneable =          110;
const int e_param_outside =        111;
const int e_file_output =          112;
const int e_domain =               113;
const int e_charRange =            114;
const int e_par_close_unbal =      115;
const int e_square_close_unbal =   116;
const int e_inv_num_format =       117;
const int e_inv_esc_sequence =     118;
const int e_eol_string =           119;
const int e_inv_token =            120;
const int e_inv_direct =           121;
const int e_byte_access =          122;
const int e_numparse_long =        123;
const int e_numparse =             124;
const int e_no_class =             125;
const int e_bitwise_op =           126;
const int e_case_decl =            127;
const int e_switch_body =          128;
const int e_select_body =          129;
const int e_default_decl =         130;
const int e_lone_end =             131;
const int e_switch_decl =          132;
const int e_select_decl =          133;
const int e_case_outside =         134;
const int e_syn_load =             135;
const int e_toplevel_func =        136;
const int e_toplevel_obj =         137;
const int e_toplevel_class =       138;
const int e_toplevel_load =        139;
const int e_syn_while  =           140;
const int e_syn_if =               141;
const int e_syn_else =             142;
const int e_syn_elif =             143;
const int e_syn_break =            144;
const int e_syn_continue =         145;
const int e_syn_for =              146;
const int e_syn_forfirst =         147;
const int e_syn_forlast =          148;
const int e_syn_formiddle =        149;
const int e_syn_try =              150;
const int e_syn_catch =            151;
const int e_syn_raise =            152;
const int e_syn_funcdecl =         153;
const int e_syn_static =           154;
const int e_syn_state =            155;
const int e_syn_launch =           156;
const int e_syn_pass =             157;
const int e_inv_const_val =        158;
const int e_syn_const =            159;
const int e_syn_export =           160;
const int e_syn_attributes =       161;
const int e_enter_notavar =        162;
const int e_syn_enter =            163;
const int e_syn_leave =            164;
const int e_syn_forin =            165;
const int e_syn_pass_in =          166;
const int e_leave_notanexp =       167;
const int e_inv_attrib =           168;
const int e_syn_class =            169;
const int e_syn_hasdef =           170;
const int e_syn_object =           171;
const int e_syn_global =           172;
const int e_syn_return =           173;
const int e_syn_arraccess =        174;
const int e_syn_funcall =          175;
const int e_syn_lambda =           176;
const int e_syn_iif =              177;
const int e_syn_dictdecl =         178;
const int e_syn_arraydecl =        179;
const int e_syn_give =             180;
const int e_syn_def =              181;
const int e_syn_fordot =           182;
const int e_syn_self_print =       183;
const int e_syn_directive =        184;
const int e_syn_import =           185;

const int e_nl_in_lit =            200;
const int e_catch_adef =           201;
const int e_catch_clash =          202;
const int e_fmt_convert =          203;
const int e_already_forfirst =     204;
const int e_already_forlast =      205;
const int e_already_formiddle =    206;
const int e_fordot_outside =       207;
const int e_interrupted =          208;
const int e_priv_access =          209;
const int e_par_unbal =            210;
const int e_square_unbal =         211;
const int e_unclosed_string =      212;
const int e_directive_unk =        213;
const int e_directive_value =      214;

const int e_open_file =            500;
const int e_loaderror =            501;
const int e_nofile =               503;
const int e_invformat =            504;
const int e_loader_unsupported =   505;

const int e_inv_params =           900;
const int e_missing_params =       901;
const int e_param_type =           902;
const int e_param_range =          903;
const int e_param_indir_code =     904;
const int e_param_strexp_code =    905;
const int e_param_fmt_code =       906;

typedef enum {
   e_orig_compiler = 1,
   e_orig_assembler = 2,
   e_orig_loader = 3,
   e_orig_vm = 4,
   e_orig_script = 5,
   e_orig_runtime = 9,
   e_orig_mod = 10
} t_origin;

class FALCON_DYN_CLASS TraceStep: public BaseAlloc
{
   String m_module;
   String m_symbol;
   uint32 m_line;
   uint32 m_pc;

public:
   TraceStep( const String &module, const String symbol, uint32 line, uint32 pc ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_pc( pc )
   {}

   const String &module() const { return m_module; }
   const String &symbol() const { return m_symbol; }
   uint32 line() const { return m_line; }
   uint32 pcounter() const { return m_pc; }

   String toString() const { String temp; return toString( temp ); }
   String &toString( String &target ) const;
};

/** Error Parameter class.
   This class provides the main Error class and its subclasses with named parameter idiom.
   Errors have many parameters and their configuration is bourdensome and also a big
   "bloaty" exactly in spots when one would want code to be small.

   This class, completely inlined, provides the compiler and the programmer with a fast
   and easy way to configure the needed parameters, preventing the other, unneded details
   from getting into the way of the coders.

   The Error class (and its subclasses) has a constructor accepting an ErrorParameter
   by reference.
   \code
      Error *e = new SomeKindOfError( ErrorParam( ... ).p1().p2()....pn() )
   \endcode

   is an acceptable grammar to create an Error.
*/

class ErrorParam: public BaseAlloc
{

public:

   /** Standard constructor.
      In the constructor a source line may be provided. This makes possible to use the
      __LINE__ ansi C macro to indicate the point in the source C++ file where an error
      is raised.
      \param code error code.
      \param line optional line where error occurs.
   */
   ErrorParam( int code, uint32 line = 0 ):
      m_errorCode( code ),
      m_line( line ),
      m_character( 0 ),
      m_pc( 0 ),
      m_sysError( 0 ),
      m_origin( e_orig_mod ),
      m_catchable( true )
      {}

   ErrorParam &code( int code ) { m_errorCode = code; return *this; }
   ErrorParam &desc( const String &d ) { m_description = d; return *this; }
   ErrorParam &extra( const String &e ) { m_extra = e; return *this; }
   ErrorParam &symbol( const String &sym ) { m_symbol = sym; return *this; }
   ErrorParam &module( const String &mod ) { m_module = mod; return *this; }
   ErrorParam &line( uint32 line ) { m_line = line; return *this; }
   ErrorParam &pc( uint32 pc ) { m_pc = pc; return *this; }
   ErrorParam &sysError( uint32 e ) { m_sysError = e; return *this; }
   ErrorParam &chr( uint32 c ) { m_character = c; return *this; }
   ErrorParam &origin( t_origin orig ) { m_origin = orig; return *this; }
   ErrorParam &hard() { m_catchable = false; return *this; }

private:
   friend class Error;

   int m_errorCode;
   String m_description;
   String m_extra;
   String m_symbol;
   String m_module;

   uint32 m_line;
   uint32 m_character;
   uint32 m_pc;
   uint32 m_sysError;

   t_origin m_origin;
   bool m_catchable;
};

/** The Error class.
   This class implements an error instance.
   Errors represent problems occoured both during falcon engine operations
   (i.e. compilation syntax errors, link errors, file I/O errors, dynamic
   library load errors ands o on) AND during runtime (i.e. VM opcode
   processing errors, falcon program exceptions, module function errors).

   When an error is raised by an engine element whith this capability
   (i.e. the compiler, the assembler, the runtime etc.), it is directly
   passed to the error handler, which has the duty to do something with
   it and eventually destroy it.

   When an error is raised by a module function with the VMachine::raiseError()
   method, the error is stored in the VM; if the error is "catchable" AND it
   occours inside a try/catch statement, it is turned into a Falcon Error
   object and passed to the script.

   When a script raises an error both explicitly via the "raise" function or
   by performing a programming error (i.e. array out of bounds), if there is
   a try/catch block at work the error is turned into a Falcon error and
   passed to the script.

   If there isn't a try/catch block or if the error is raised again by the
   script, the error instance is passed to the VM error handler.

   Scripts may raise any item, which may not necessary be Error instances.
   The item is then copied in the m_item member and passed to the error
   handler.
*/

class FALCON_DYN_CLASS Error: public BaseAlloc
{
protected:
   int m_errorCode;
   String m_description;
   String m_extra;
   String m_symbol;
   String m_module;
   String m_className;

   uint32 m_line;
   uint32 m_character;
   uint32 m_pc;
   uint32 m_sysError;

   uint32 m_refCount;

   t_origin m_origin;
   bool m_catchable;
   Item m_raised;

   List m_steps;
   ListElement *m_stepIter;

   Error *m_nextError;
   Error *m_LastNextError;

   /** Empty constructor.
      The error must be filled with proper values.
   */
   Error( const String &className ):
      m_errorCode ( e_none ),
      m_sysError( 0 ),
      m_line( 0 ),
      m_character( 0 ),
      m_pc( 0 ),
      m_origin( e_orig_runtime ),
      m_refCount( 1 ),
      m_className( className ),
      m_catchable( true ),
      m_nextError( 0 ),
      m_LastNextError( 0 )
   {
      m_raised.setNil();
   }

   /** Copy constructor. */
   Error( const Error &e );

   /** Minimal constructor.
      If the description is not filled, the toString() method will use the default description
      for the given error code.
   */
   Error( const String &className, const ErrorParam &params ):
      m_errorCode ( params.m_errorCode ),
      m_sysError( params.m_sysError ),
      m_line( params.m_line ),
      m_character( params.m_character ),
      m_pc( params.m_pc ),
      m_origin( params.m_origin ),
      m_refCount( 1 ),
      m_className( className ),
      m_catchable( params.m_catchable ),
      m_description( params.m_description ),
      m_extra( params.m_extra ),
      m_symbol( params.m_symbol ),
      m_module( params.m_module ),
      m_nextError( 0 ),
      m_LastNextError( 0 )
   {
      m_raised.setNil();
   }

   /** Private destructor.
      Can be destroyed only via decref.
   */
   virtual ~Error();
public:


   void errorCode( int ecode ) { m_errorCode = ecode; }
   void systemError( uint32 ecode ) { m_sysError = ecode; }
   void errorDescription( const String &errorDesc ) { m_description = errorDesc; }
   void extraDescription( const String &extra ) { m_extra = extra; }
   void module( const String &moduleName ) { m_module = moduleName; }
   void symbol( const String &symbolName )  { m_symbol = symbolName; }
   void line( uint32 line ) { m_line = line; }
   void character( uint32 chr ) { m_character = chr; }
   void pcounter( uint32 pc ) { m_pc = pc; }
   void origin( t_origin o ) { m_origin = o; }
   void catchable( bool c ) { m_catchable = c; }
   void raised( const Item &itm ) { m_raised = itm; }

   int errorCode() const { return m_errorCode; }
   uint32 systemError() const { return m_sysError; }
   const String &errorDescription() const { return m_description; }
   const String &extraDescription() const { return m_extra; }
   const String &module() const { return m_module; }
   const String &symbol() const { return m_symbol; }
   uint32 line() const { return m_line; }
   uint32 character() const { return m_character; }
   uint32 pcounter() const { return m_pc; }
   t_origin origin() const { return m_origin; }
   bool catchable() const { return m_catchable; }
   const Item &raised() const { return m_raised; }

   String toString() const { String temp; return toString( temp ); }
   virtual String &toString( String &target ) const;

   /** Writes only the heading of the error to the target string.
      The error heading is everything of the error without the traceback.
      This method never recurse on error lists; only the first heading is returned.
      \note the input target string is not cleared; error contents are added at
         at the end.
      \note The returned string doesn't terminate with a "\n".
   */
   virtual String &heading( String &target ) const;


   void appendSubError( Error *sub );

   /** Returns an object that can be set in a Falcon item and handled by a script.
      This method converts the error object in a Falcon Object, derived from the
      proper class.

      The method must be fed with a virtual machine. The target virtual machine
      should have linked a module providing a "specular class". This method will
      search the given VM for a class having the same name as the one that is
      returned by the className() method (set in the constructor by the subclasses
      of Error), and it will create an instance of that class. The method
      will then fill the resulting object with the needed values, and finally
      it will set itself as the User Data of the given object.

      The target class Falcon should be a class derived from the Core class "Error",
      so that the inherited methods as "toString" and "traceback" are inherited too,
      and so that a check on "Error" inheritance will be positive.

   */
   virtual CoreObject *scriptize( VMachine *vm );

   void addTrace( const String &module, const String &symbol, uint32 line, uint32 pc );
   bool nextStep( String &module, String &symbol, uint32 &line, uint32 &pc );
   void rewindStep();

   const String &className() const { return m_className; }

   void incref();
   void decref();
};



class GenericError: public Error
{
public:
   GenericError():
      Error( "Error" )
   {}

   GenericError( const ErrorParam &params  ):
      Error( "Error", params )
      {}
};

class CodeError: public Error
{
public:
   CodeError():
      Error( "CodeError" )
   {}

   CodeError( const ErrorParam &params  ):
      Error( "CodeError", params )
      {}
};

class SyntaxError: public Error
{
public:
   SyntaxError():
      Error( "SyntaxError" )
   {}

   SyntaxError( const ErrorParam &params  ):
      Error( "SyntaxError", params )
      {}
};

class RangeError: public Error
{
public:
   RangeError():
      Error( "RangeError" )
   {}

   RangeError( const ErrorParam &params  ):
      Error( "RangeError", params )
      {}
};

class MathError: public Error
{
public:
   MathError():
      Error( "MathError" )
   {}

   MathError( const ErrorParam &params  ):
      Error( "MathError", params )
      {}
};

class TypeError: public Error
{
public:
   TypeError():
      Error( "TypeError" )
   {}

   TypeError( const ErrorParam &params  ):
      Error( "TypeError", params )
      {}
};

class IoError: public Error
{
public:
   IoError():
      Error( "IoError" )
   {}

   IoError( const ErrorParam &params  ):
      Error( "IoError", params )
      {}
};


class ParamError: public Error
{
public:
   ParamError():
      Error( "ParamError" )
   {}

   ParamError( const ErrorParam &params  ):
      Error( "ParamError", params )
      {}
};

class ParseError: public Error
{
public:
   ParseError():
      Error( "ParseError" )
   {}

   ParseError( const ErrorParam &params  ):
      Error( "ParseError", params )
      {}
};

class CloneError: public Error
{
public:
   CloneError():
      Error( "CloneError" )
   {}

   CloneError( const ErrorParam &params  ):
      Error( "CloneError", params )
      {}
};

class InterruptedError: public Error
{
public:
   InterruptedError():
      Error( "InterruptedError" )
   {}

   InterruptedError( const ErrorParam &params  ):
      Error( "InterruptedError", params )
      {}
};



class FALCON_DYN_CLASS ErrorCarrier: public UserData
{
   Error *m_error;
   String m_origin;

public:
   ErrorCarrier( Error *carried );

   virtual ~ErrorCarrier();
   virtual bool isReflective();
   virtual void getProperty( const String &propName, Item &prop );
   virtual void setProperty( const String &propName, Item &prop );

   Error *error() const { return m_error; }
};

/** Returns the description of a falcon error.
   In case the error ID is not found, a sensible message will be returned.
*/
const String &errorDesc( int errorCode );

namespace core {
FALCON_FUNC_DYN_SYM Error_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM SyntaxError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM CodeError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM IoError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM RangeError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM MathError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM ParamError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM ParseError_init ( ::Falcon::VMachine *vm );
}

}

#endif

/* end of error.h */
