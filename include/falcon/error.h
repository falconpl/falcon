/*
   FALCON - The Falcon Programming Language.
   FILE: error.h

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ERROR_H
#define FALCON_ERROR_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/enumerator.h>
#include <falcon/tracestep.h>
#include <falcon/traceback.h>
#include <falcon/classes/classerror.h>

namespace Falcon {

class Error;
class Class;
class Error_p;
class VMContext;

// Declare the error IDS
#define FLC_DECLARE_ERROR_TABLE
#include <falcon/error_messages.h>
#undef FLC_DECLARE_ERROR_TABLE


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

class ErrorParam
{

public:
   typedef enum {
      e_orig_unknown = 0,
      e_orig_compiler = 1,
      e_orig_linker = 2,
      e_orig_loader = 3,
      e_orig_vm = 4,
      e_orig_script = 5,
      e_orig_runtime = 9,
      e_orig_mod = 10
   } t_origin;

   /** Standard constructor.
      In the constructor a source line may be provided. This makes possible to use the
      __LINE__ ansi C macro to indicate the point in the source C++ file where an error
      is raised.

    Similarly, the file is parameter can be set to __FILE__.

      \param code error code.
      \param file the file where the error is raised.
      \param line optional line where error occurs.
   */
   ErrorParam( int code, uint32 signLine = 0, const char* file = 0 ):
      m_errorCode( code ),
      m_module(),
      m_line( 0 ),
      m_chr( 0 ),
      m_sysError( 0 ),
      m_origin( e_orig_mod ),
      m_catchable( true )
      {
         if( file != 0 ) {
            m_signature = file;
            if( signLine != 0 ) {
               m_signature.A(":").N(signLine);
            }
            m_signature.bufferize();
         }
      }

   ErrorParam( int code, uint32 signLine, const String& signature ):
      m_errorCode( code ),
      m_line( 0 ),
      m_chr( 0 ),
      m_sysError( 0 ),
      m_origin( e_orig_mod ),
      m_catchable( true )
      {
         m_signature = signature;
         if( signLine > 0 )
            m_signature.A(":").N(signLine);
         else
         {
            m_signature.bufferize();
         }
      }
   
   ErrorParam():
      m_errorCode( 0 ),
      m_module( "" ),
      m_line( 0 ),
      m_chr( 0 ),
      m_sysError( 0 ),
      m_origin( e_orig_mod ),
      m_catchable( true )
      {}

   /**
    * Fill an error with the current context.
    *
    * This extracts function, module, line and execution status
    * from the context, and uses them to configure the error parameters.
    *
    * \note For convenience, the definition of this constructor is in vmcontext.cpp
    */
   ErrorParam( int code, VMContext* ctx, const char* file=0, int signLine = 0 );

   ErrorParam &code( int code ) { m_errorCode = code; return *this; }
   ErrorParam &desc( const String &d ) { m_description = d; return *this; }
   ErrorParam &extra( const String &e ) { m_extra.bufferize(e); return *this; }
   ErrorParam &symbol( const String &sym ) { m_symbol = sym; return *this; }
   ErrorParam &module( const String &mod ) { m_module = mod; return *this; }
   ErrorParam &path( const String &p ) { m_path = p; return *this; }
   ErrorParam &line( uint32 line ) { m_line = line; return *this; }
   ErrorParam &chr( uint32 chr ) { m_chr = chr; return *this; }
   ErrorParam &sysError( uint32 e ) { m_sysError = e; return *this; }
   ErrorParam &origin( t_origin orig ) { m_origin = orig; return *this; }
   ErrorParam &hard() { m_catchable = false; return *this; }
   ErrorParam &sign( const String& str ) { m_signature = str; return *this; }

private:
   friend class Error;

   int m_errorCode;
   String m_description;
   String m_extra;
   String m_symbol;
   String m_module;
   String m_path;
   String m_signature;

   uint32 m_line;
   uint32 m_chr;
   uint32 m_sysError;

   t_origin m_origin;
   bool m_catchable;
};


/** The Error class.
 *
   This class implements an error instance.

   Errors represent problems occurred both during falcon engine operations
   (i.e. compilation syntax errors, link errors, file I/O errors, dynamic
   library load errors ands o on) AND during runtime (i.e. VM opcode
   processing errors, falcon program exceptions, module function errors).

   As errors can be handled directly and eventually created by both C++ code
   and falcon script code, each Error subclass is required to provide an error
   handler; as usual, the Error instance is seen as a Falcon instance, and its
   handler is called ClassXXX, where XXX is the name of the instance class.

   For example, a MathError instance requires a ClassMathError to be defined for
   the scripts to use it on need. When a script invokes a MathError() constructor,
   to create and eventually throw a math error, the ClassMathError() handler will
   generate a C++ MathError* instance.

   The two entities can travel coupled or be de-coupled when needed. For instance,
   suppose a script terminates because of a division by zero error. The item
   traveling in the process will be a pair of (ClassMathError*, MathError*). As the
   process terminates with error, the process owner may receive the MathError that was
   thrown in several ways; for instance, in some cases it might be thrown again at C++
   level and eventually handled in a C++ clause.

   The reverse operation is also possible: engine related or third party code can
   generate a Error subclass instance, and if it traverses the VM, it can be caught
   by script handlers, and turned in a pair of (ClassXXX*, XXX*).

   The link between an Error class and its ClassError handler is kept via a pair of methods:
   the Error::handler() method on the error side, the Class::createInstance() on the
   Class handler side.

   To help developers in the task of accessing a single instance of a ClassError subclass that is
   used as handler for multiple instances of the related Error subclass, the engine provides
   a error class registration feature. Although optional, it is useful in most of the
   possible usage contexts.

\section error_registration

    Registering a ClassError handler with the engine is done by invoking
    Engine::instance()->registerError( Class* ) (or letting the ClassError
    constructor to do this automatically).

    The instances are then available by their declared name, that is, by the name
    that is visible to the scripts, through the Engine::getError() method.

     @note The declared name may include a default namespace given in the name, for instance
       "MyExtension.MyErrorClass".

    This mechanism is exploited by the base version of the Error::handler() method.
    Its default behavior is that that of returning a mantra in the
    engine having the same name of the given error. In other words:

    @code
      // default operations involved in error framing.
      Error* someError = generateError();
      Class* someErrorHandler = Engine::instance()->getError( someError->name() );
      Item scriptLevelError = Item( someErrorHandler, someError );
    @endcode

    This behavior is just a basic pattern that is used mainly for engine and feather modules
    error classes, although embedding and third party modules are free and encouraged to
    use this.

    @note Registering the Class as a error handler doesn't automatically adds the class
    as an engine-level visible (superglobal) mantra. Normal mantra declaration, registration
    and visibility rules still apply. Also, engine-level error registration doesn't reference
    or keeps the mantra alive in any way; the Class handler should unregister itself when it
    goes out of scope.

*/

class FALCON_DYN_CLASS Error
{
public:

   /** Enumerator for sub-errors.
    @see enumerateSuberrors
    */
   typedef Enumerator<Error> ErrorEnumerator;

   /** Sets the error code.
    \param ecode an error ID.
    */
   void errorCode( int ecode ) { m_errorCode = ecode; }
   /** Sets the system error code.

    Many errors are raised after system errors in I/O operations.
    This is a useful fields that avoids the need to recast or use ad-hoc
    strucures for I/O or system related errors.
    \param ecode The system error that caused this error to be raised.
    */
   void systemError( uint32 ecode ) { m_sysError = ecode; }
   void errorDescription( const String &errorDesc ) { m_description = errorDesc; }
   void extraDescription( const String &extra ) { m_extra = extra; }
   void module( const String &moduleName ) { m_module = moduleName; }
   void path( const String &path ) { m_path = path; }
   void mantra( const String &symbolName )  { m_mantra = symbolName; }
   void line( int32 line ) { m_line = line; }
   void chr( int32 chr ) { m_chr = chr; }
   void origin( ErrorParam::t_origin o ) { m_origin = o; }
   void catchable( bool c ) { m_catchable = c; }
   void raised( const Item &itm ) { m_raised = itm; m_bHasRaised = true; }
   void sign( const String& s ) { m_signature = s; }

   int errorCode() const { return m_errorCode; }
   uint32 systemError() const { return m_sysError; }
   const String &errorDescription() const { return m_description; }
   const String &extraDescription() const { return m_extra; }
   const String &module() const { return m_module; }
   const String &path() const { return m_path; }
   const String &mantra() const { return m_mantra; }
   int32 line() const { return m_line; }
   int32 chr() const { return m_chr; }
   ErrorParam::t_origin origin() const { return m_origin; }
   bool catchable() const { return m_catchable; }
   const Item &raised() const { return m_raised; }
   bool hasRaised() const { return m_bHasRaised; }
   const String& signature() const { return m_signature; }

   inline String describe( bool bAddPath=false, bool bAddParams=false, bool bAddSign=false ) const {
      String s; describeTo(s, bAddPath, bAddParams, bAddSign ); return s;
   }
   /** Renders the error to a string.
    */
   virtual void describeTo( String &target, bool bAddPath=false, bool bAddParams=false, bool bAddSign=false ) const;

   /** Writes only the heading of the error to the target string.
      The error heading is everything of the error without the traceback.
      This method never recurse on error lists; only the first heading is returned.
      \note the input target string is not cleared; error contents are added at
         at the end.
      \note The returned string doesn't terminate with a "\n".
   */
   virtual String &heading( String &target ) const;

   /** Renders the full code of this error.
    * \param target The target string where to place the result.
    * \return the target string;
    *
    * A complete error code is formed with a two-letter code indicating the
    * error origin, and a 0-padded 4 cyphers numeric code.
    *
    * \note The \b target string is used additively (it is not cleared when the description is added).
    */
   String& fullCode( String& target ) const;

   /** Renders the location (module, function, line and character) of this error.
    * \param target The target string where to place the result.
    * \return the target string;
    * \note The \b target string is used additively (it is not cleared when the description is added).
    */
   String& location( String& target ) const;

   /** Render the generator errors.
    * \param target the string where to add the errors.
    * \param addSignature If true, the signature field of the sub-errors is added to their description.
    * \return the target string
    *
    * \note The \b target string is used additively (it is not cleared when the description is added).
    */
   String& describeSubErrors( String& target, bool bAddPath=false, bool bAddParams=false, bool bAddSign=false ) const;

   /** Render the trace.
    * \param target the string where to add the errors.
    * \param bAddPath If true, add the source module URI specification to each line of the trace.
    * \param bAddParams If true, add the parameters to the call list.
    * \return the target string
    *
    * \note The \b target string is used additively (it is not cleared when the description is added).
    */
   String& describeTrace( String& target, bool bAddPath = false, bool bAddParams = false ) const;

   /** Renders the two-characters origin code for this error.
    *
    * \param target The target string where to place the result.
    * \return the target string.
    *
    * The origin code of the error is a two-letter code that indicates
    * which part of the Falcon system generated the error. It can be one
    * of the following:
    *
    * - Compiler: "CP"
      - Linker: "LK";
      - Loader: "LD"
      - Virtual Machine: "VM"
      - Runtime: "RT"
      - Module/extension: "MD"
      - Script/source Falcon code: "SC"

    * \note The \b target string is used additively (it is not cleared when the description is added).
    */
   String& originCode(String& target ) const;

   /** Adds a sub-error to this error.

    Some errors store multiple errors that cause a more general error condition.
    For example, a compilation may fail due to multiple syntax errors. This fact
    is represented by raising a CompileError which contains all the errors that
    caused the compilation to fail.

    */
   void appendSubError( Error *sub );

   /** Creates a falcon instance that may be used directly by a script.

    The error is referenced and stored in the data field of the item, and
    the handler class is set to the scriptClass that was set when creating
    the instance.

    This makes the item immediately useable from the script.
   */
   void scriptize( Item& tgt );

   const Class* handler() const;
   void handler( const Class* ) const;

   /**
    * Returns a sub-class specific description of the error code.
    *
    * The base class version describes the engine error codes (allocated
    * under 1000).
    *
    * Overriding the base version allows to return consistent descriptions
    * for given user-specific error codes.
    *
    * The macro FALCON_DECLARE_ERROR_INSTANCE_WITH_DESC allows to declare
    * a subclass and provide some description, via the FALCON_ERROR_CLASS_DESC macro,
    * like in this example:
    *
    * \code
    * FALCON_DECLARE_ERROR_INSTANCE_WITH_DESC( MyError,
    *    FALCON_ERROR_CLASS_DESC( 10001, "Some error condition" )
    *    FALCON_ERROR_CLASS_DESC( 10002, "Some other error condition" )
    * )
    * FALCON_DECLARE_ERROR_CLASS( MyError )
    * \endcode
    *
    * \note The description() field overrides the values returned by
    * this method.
    */
   virtual void describeErrorCodeTo( int errorCode, String& tgt ) const;

   inline String describeErrorCode( int errorCode ) const
   {
      String temp; describeErrorCodeTo( errorCode, temp ); return temp;
   }

   /** Enumerate the sub-errors.
    \param rator A ErrorEnumerator that is called back with each sub-error in turn.
    \see appendSubError
    */
   void enumerateErrors( ErrorEnumerator &rator ) const;

   /** Return the name of this error class.
    *  Set in the constructcor.
    */
   const String &className() const{ return m_name; }
   
   /** Gets the first sub-error.
    Some errors are used to wrap a single lower level error. For example,
    a virtual machine may be terminated because an error was raised and
    nothing caught it; in that case, the termination error is UncaughtError,
    and it will "box" the error that was raised originally inside the script.

    This is called error "boxing". This method allows to access the first
    sub-error, that may be the boxed error, without the need to setup an
    enumerator callback.
    \see appendSubError.
    \return The boxed error, or 0 if this error isn't boxing anything.
    */
   Error* getBoxedError() const;

   /** Return true if this error has been filled with a traceback.*/
   bool hasTraceback() const { return m_tb != 0; }

   bool hasSubErrors() const;

   /** Increment the reference count of this object */
   void incref() const;

   /** Decrements the reference count of this object.
    The error must be considered invalid after this call.
    */
   void decref();

   /** Sets all de values in the error structure. */
   void set( const ErrorParam& params );
   
   /** Gest the traceback associated with this error */
   const TraceBack* traceBack() const { return m_tb; }

   /** Sets the traceback associated with this error */
   void setTraceBack( TraceBack* tb ){
      delete m_tb;
      m_tb = tb;
   }

protected:

   Error( const String& name, const ErrorParam &params );

   /** Minimal constructor.
      If the description is not filled, the toString() method will use the default description
      for the given error code.
   */
   Error( const String& name );

   Error( const Class* handler );
   Error( const Class* handler, const ErrorParam &params );

   mutable atomic_int m_refCount;

   int m_errorCode;
   String m_description;
   String m_extra;
   String m_mantra;
   String m_module;
   String m_path;
   String m_signature;

   int32 m_line;
   int32 m_chr;
   uint32 m_sysError;


   ErrorParam::t_origin m_origin;
   bool m_catchable;
   Item m_raised;
   bool m_bHasRaised;

   String m_name;

protected:
   /** Private destructor.
      Can be destroyed only via decref.
   */
   virtual ~Error();

private:
   Error_p* _p;
   TraceBack* m_tb;
   mutable const Class* m_handler;
};

}

#define FALCON_DECLARE_ERROR_CLASS( __name__ ) \
         FALCON_DECLARE_ERROR_CLASS_EX( __name__, )

#define  FALCON_DECLARE_ERROR_INSTANCE( __name__ ) \
         FALCON_DECLARE_ERROR_INSTANCE_WITH_DESC( __name__, );

/** Macro used to declare an error class and it's related handler class.
 * \param __name__ The name of the error class as seen from the script.
 *
 * This macro will define Class<name> and <name> classes, with minimal
 * method override to make them functional.
 */
#define FALCON_DECLARE_ERROR( __name__ ) \
   FALCON_DECLARE_ERROR_INSTANCE( __name__ ) \
   FALCON_DECLARE_ERROR_CLASS( __name__ )

/** Macro used to declare a system level error class and it's related handler class.
 * \param __name__ The name of the error class as seen from the script.
 *
 * This macro will define Class<name> and <name> classes, with minimal
 * method override to make them functional.
 *
 * Also, the ClassXXX handler will be automatically placed in the engine,
 * becoming a superglobal available to every Falcon script run in this
 * engine.
 *
 */
#define FALCON_DECLARE_SYS_ERROR( __name__ ) \
   FALCON_DECLARE_ERROR_INSTANCE( __name__ ) \
   FALCON_DECLARE_ERROR_CLASS_EX( __name__, ::Falcon::Engine::instance()->addMantra(::Falcon::Engine::instance()->registerError(this)) )

/**
 * Creates a signed error.
 *
 * This macro creates an instance of a class already configured with
 * the current context and signed at given position.
 *
 * The macro uses the __LINE__ standard C99 macro to determine the
 * current file line, and the SRC macro which is Falcon specific
 * and gets defined as __FILE__ by falcon/setup.h, unless
 * it's specifically set in the source file before its inclusion
 *
 * @note Remember to add the ';' after the macro.
 */
#define FALCON_SIGN_ERROR( Error_Class__, error_code__ ) \
         (new Error_Class__(ErrorParam(error_code__, __LINE__, SRC  ).line(-1) ))

/**
 * More compact error macro.
 *
 * This is like FALCON_SIGN_ERROR, but adds a call to the VMContext::raiseError
 * method and dereferences the returned error immediately.
 */
#define FALCON_RESIGN_ERROR( Error_Class__, error_code__, VMContext__ ) \
         (VMContext__->raiseError(new Error_Class__->(ErrorParam(error_code__, __LINE__, SRC ).line(-1) ))->decref())

/**
 * Creates a signed error with extra information.
 *
 * This macro creates an instance of a class already configured with
 * the current context and signed at given position.
 *
 * The extra parameter is directly expanded after the ErrorParam()
 * call, so it must be a sequence of self-returning methods.
 * For instance, to create a signed error and specify the system
 * error and the extra description, use
 *
 * @code
 * throw FALCON_SIGN_XERROR( IOError, e_io_code, vmcontext,
 *       .extra("Can't open this file")
 *       .sysError(15)
 *       );
 * @endcode
 *
 * The macro uses the __LINE__ standard C99 macro to determine the
 * current file line, and the SRC macro which is Falcon specific
 * and gets defined as __FILE__ by falcon/setup.h, unless
 * it's specifically set in the source file before its inclusion.
 *
 *@note Remember to add the ';' after the macro.
 */
#define FALCON_SIGN_XERROR( Error_Class__, error_code__, Extra__ ) \
         (new Error_Class__(ErrorParam(error_code__, __LINE__, SRC ).line(-1) Extra__ ))


/**
 * More compact error macro.
 *
 * This is like FALCON_SIGN_XERROR, but adds a call to the VMContext::raiseError
 * method and dereferences the returned error immediately
 */
#define FALCON_RESIGN_XERROR( Error_Class__, error_code__, VMContext__, Extra__ ) \
         VMContext__ = VMContext__;\
         throw new Error_Class__(ErrorParam(error_code__, __LINE__, SRC ).line(-1) Extra__ );

         //(VMContext__->raiseError(new Error_Class__(ErrorParam(error_code__, __LINE__, SRC ) Extra__ ))->decref())
/**
 * Prepare a read-only-property error on the given property
 */

#define FALCON_SIGN_ROPROP_ERROR( prop ) \
      (new AccessError(ErrorParam(e_prop_ro, __LINE__, SRC  ).extra(prop) ))

/**
 * Prepare a read-only-property error on the given property
 */
#define FALCON_RESIGN_ROPROP_ERROR( prop, VMContext__ ) \
         (VMContext__->raiseError(new AccessError(ErrorParam(e_prop_ro, __LINE__, SRC  ).extra(prop) ))->decref())


#define FALCON_DECLARE_ERROR_CLASS_EX( __name__, __extra__ ) \
   class Class##__name__: public ::Falcon::ClassError \
   {\
   public:\
      inline Class##__name__(): ::Falcon::ClassError( #__name__ ) { setParent( Engine::instance()->getError("Error")); __extra__; } \
      inline Class##__name__( bool bInEngine ): ::Falcon::ClassError( #__name__, bInEngine ) { setParent( Engine::instance()->getError("Error"));  __extra__; } \
      inline virtual ~Class##__name__(){} \
      inline virtual void* createInstance() const { return new __name__(this); } \
      Error* createError( const ErrorParam& params ) const {return new __name__(this, params); }\
   };\
   extern Class##__name__* __name__##_handler;


#define  FALCON_DECLARE_ERROR_INSTANCE_WITH_DESC( __name__, __DESC__ ) \
   class __name__ : public ::Falcon::Error\
   {\
   public:\
      __name__ (): ::Falcon::Error( #__name__ ) {} \
      __name__ ( const ErrorParam& ep ): ::Falcon::Error( #__name__, ep ) {} \
      __name__ ( const Class* handler ): ::Falcon::Error( handler ) {} \
      __name__ ( const Class* handler, const ErrorParam& ep ): ::Falcon::Error( handler, ep ) {} \
       inline virtual ~__name__() {}\
       inline virtual void describeErrorCodeTo( int errorCode, String& tgt ) const \
         {\
             switch(errorCode) {\
             case -100: break;\
             __DESC__\
             default:  Error::describeErrorCodeTo(errorCode,tgt); break;\
             }\
         }\
   };


#define FALCON_ERROR_CLASS_DESC( __id__, __desc__ )  case __id__: tgt = __desc__; break;

#endif   /* FALCON_ERROR_H */

/* end of error.h */
