/*
   FALCON - The Falcon Programming Language.
   FILE: error.h

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom feb 18 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Error class definition file.
   (this file contains also the TraceStep class).
*/

#ifndef FALCON_ERROR_H
#define FALCON_ERROR_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>
#include <falcon/genericlist.h>
#include <falcon/string.h>
#include <falcon/crobject.h>
#include <falcon/reflectfunc.h>

namespace Falcon {

class Error;

namespace core {
FALCON_FUNC_DYN_SYM Error_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM SyntaxError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM CodeError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM IoError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM AccessError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM MathError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM ParamError_init ( ::Falcon::VMachine *vm );
FALCON_FUNC_DYN_SYM ParseError_init ( ::Falcon::VMachine *vm );

/** Reflective function to support error property: code */
extern reflectionFuncDecl Error_code_rfrom;
extern reflectionFuncDecl Error_description_rfrom;
extern reflectionFuncDecl Error_message_rfrom;
extern reflectionFuncDecl Error_systemError_rfrom;
extern reflectionFuncDecl Error_origin_rfrom;
extern reflectionFuncDecl Error_module_rfrom;
extern reflectionFuncDecl Error_symbol_rfrom;
extern reflectionFuncDecl Error_line_rfrom;
extern reflectionFuncDecl Error_pc_rfrom;
extern reflectionFuncDecl Error_subErrors_rfrom;

extern reflectionFuncDecl Error_code_rto;
extern reflectionFuncDecl Error_description_rto;
extern reflectionFuncDecl Error_message_rto;
extern reflectionFuncDecl Error_systemError_rto;
extern reflectionFuncDecl Error_origin_rto;
extern reflectionFuncDecl Error_module_rto;
extern reflectionFuncDecl Error_symbol_rto;
extern reflectionFuncDecl Error_line_rto;
extern reflectionFuncDecl Error_pc_rto;

/** Reflective class for error */
class ErrorObject: public CRObject
{
public:
   ErrorObject( const CoreClass* cls, Error *err );
   Error* getError() const { return (::Falcon::Error*) getUserData(); }

   virtual ~ErrorObject();
   virtual void gcMark( uint32 mark );
   virtual ErrorObject *clone() const;
};

CoreObject* ErrorObjectFactory( const CoreClass *cls, void *user_data, bool bDeserial );
}


// Declare the messaages...
#include <falcon/eng_messages.h>

// and set the error IDS.
#define FLC_DECLARE_ERROR_TABLE
#include <falcon/eng_messages.h>
#undef FLC_DECLARE_ERROR_TABLE

typedef enum {
   e_orig_unknown = 0,
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
   String m_modpath;

public:
   //TODO: Remove this version in the next version.
   TraceStep( const String &module, const String symbol, uint32 line, uint32 pc ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_pc( pc )
   {}

   TraceStep( const String &module, const String &mod_path, const String symbol, uint32 line, uint32 pc ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_pc( pc ),
      m_modpath( mod_path )
   {}

   const String &module() const { return m_module; }
   const String &modulePath() const { return m_modpath; }
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
   ErrorParam &extra( const String &e ) { m_extra.bufferize(e); return *this; }
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
   int32 m_refCount;

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
      m_refCount( 1 ),
      m_errorCode ( e_none ),
      m_className( className ),
      m_line( 0 ),
      m_character( 0 ),
      m_pc( 0 ),
      m_sysError( 0 ),
      m_origin( e_orig_unknown ),
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
      m_refCount( 1 ),
      m_errorCode ( params.m_errorCode ),
      m_description( params.m_description ),
      m_extra( params.m_extra ),
      m_symbol( params.m_symbol ),
      m_module( params.m_module ),
      m_className( className ),
      m_line( params.m_line ),
      m_character( params.m_character ),
      m_pc( params.m_pc ),
      m_sysError( params.m_sysError ),
      m_origin( params.m_origin ),
      m_catchable( params.m_catchable ),
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

   Error():
      m_refCount( 1 ),
      m_errorCode ( e_none ),
      m_className( "Error" ),
      m_line( 0 ),
      m_character( 0 ),
      m_pc( 0 ),
      m_sysError( 0 ),
      m_origin( e_orig_unknown ),
      m_catchable( true ),
      m_nextError( 0 ),
      m_LastNextError( 0 )
   {
      m_raised.setNil();
   }

   Error( const ErrorParam &params ):
      m_refCount( 1 ),
      m_errorCode ( params.m_errorCode ),
      m_description( params.m_description ),
      m_extra( params.m_extra ),
      m_symbol( params.m_symbol ),
      m_module( params.m_module ),
      m_className( "Error" ),
      m_line( params.m_line ),
      m_character( params.m_character ),
      m_pc( params.m_pc ),
      m_sysError( params.m_sysError ),
      m_origin( params.m_origin ),
      m_catchable( params.m_catchable ),
      m_nextError( 0 ),
      m_LastNextError( 0 )
   {
      m_raised.setNil();
   }

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
   void addTrace( const String &module, const String &mod_path, const String &symbol, uint32 line, uint32 pc );
   bool nextStep( String &module, String &symbol, uint32 &line, uint32 &pc );
   void rewindStep();

   const String &className() const { return m_className; }

   void incref();
   void decref();

   Error* subError() const { return m_nextError; }

   virtual Error *clone() const;

   bool hasTraceback() const { return ! m_steps.empty(); }
};



class GenericError: public Error
{
public:
   GenericError():
      Error( "GenericError" )
   {}

   GenericError( const ErrorParam &params  ):
      Error( "GenericError", params )
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

class AccessError: public Error
{
public:
   AccessError():
      Error( "AccessError" )
   {}

   AccessError( const ErrorParam &params  ):
      Error( "AccessError", params )
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

class MessageError: public Error
{
public:
   MessageError():
      Error( "MessageError" )
   {}

   MessageError( const ErrorParam &params  ):
      Error( "MessageError", params )
      {}
};

class TableError: public Error
{
public:
   TableError():
      Error( "TableError" )
   {}

   TableError( const ErrorParam &params  ):
      Error( "TableError", params )
      {}
};


/** Returns the description of a falcon error.
   In case the error ID is not found, a sensible message will be returned.
*/
const String &errorDesc( int errorCode );


}

#endif

/* end of error.h */
