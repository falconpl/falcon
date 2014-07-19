/*
   FALCON - The Falcon Programming Language.
   FILE: function.h

   Function objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FUNCTION_H_
#define FALCON_FUNCTION_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/symbolmap.h>
#include <falcon/mantra.h>

namespace Falcon
{

class Collector;
class VMContext;
class Error;
class Class;
class Item;
class ClassFunction;
class Closure;
class TextWriter;

/**
 Falcon function.

 This class represents the minimal execution unit in Falcon. It's a set of
 code (to be excuted), symbols (parameters, local variables and reference to
 global variables in the module) and possibly closed values.

 Functions can be directly executed by the virtual machine.

 They usually reside in a module, of which they are able to access the global
 variable vector (and of which they keep a reference).

 To achieve higher performance, functions are not treated as
 normal garbageable items (the vast majority of them is never really
 destroyed). They become garbageable when their module is explicitly
 unloaded while linked, or when they are created dynamically as closures,
 or when constructed directly by the code.

 Functions can be created by modules or directly from the code. In this case,
 they aren't owned by any module and never considered as collectible.
 
*/

class FALCON_DYN_CLASS Function: public Mantra
{
public:
   Function( const String& name, Module* owner = 0, int32 line = 0 );
   virtual ~Function();

   /** Parses the description of the function.
    \param dsec Descriptive list of parameters and signature.
    
    This method parses the description of a Falcon function, adding the parameter
    names and the signature that are needed for dynamic parameter binding,
    documentation, automated checks and error reporting.
    
    For instance, this code:
    \code
    Function f( "afunc" );
    f.addParam( "param0" );
    f.addParam( "param1" );
    f.signature( "S,[N]" );
    \endcode
    
    and this:
    \code
    Function f( "afunc" );
    f.parseDescription( "param0:S,param1:[N]" );
    \endcode
    
    are equivalent.
    
    The description may start with "&" character. "&" means that
    the function is ETA (by default, functions are non-eta).
    
    \return false on malformed parameter string, 
    */
   bool parseDescription( const String& desc );
   
   /** Return the parameter list and their signature in canonical format.
    *
    *  This method calculates the description of the function parameters
    *  as accepted by parseDescription(). It's the list of the paramters
    *  associated with their signature type.
    *
    *  @note Performance warning: the description is not stored, and calculated at each request.
    *  Also, the string is returned by copy.
    */
   String getDescription() const;
   
   //void getParams( int pCount,  ... );

   //void methodOf( Class* cls ) { m_methodOf = cls; }
   void methodOf( Class* cls );
   Class* methodOf() const { return m_methodOf; }

   virtual String fullName() const;

   /** Sets the signature of the function.
    \param sign A string with the expected parameters of the function.
    */
   void signature( const String& sign ) { m_signature = sign; }

   /** Gets the signature of the function.
    \return A string representing the expected parameters of the function.
    */
   const String& signature() const { return m_signature; }

   /** Calculates the signature removing the first parameter.
     This is used by methodic functions to calculate the signature of the method version.
    */
   String methodicSignature() const;

   Error* paramError( int line = 0, const char* modName = 0, bool methodic=false ) const;
   Error* paramError( const String& extra, int line = 0, const char* modName = 0 ) const;

   /** Executes the call.
    \param ctx The Virtual Machine context on which the function is executed.
    \param pCount Number of parameters in the stack for this function.
    
    The call execution may be either immediate or deferred; for example,
    the call may just leaves PSteps to be executed by the virtual machine.

    In case of deferred calls, invoke() must also push proper return PStep codes.
    In case of immediate calls, invoke() must also perform the return frame
    code in the virtual machine by calling VMcontext::returnFrame().

    To "return" a value to the caller, set the value of the VMcontext::topData()
    item after invoking the return frame, or use the 
    VMcontext::returnFrame(const Item&) version.
    */
   virtual void invoke( VMContext* ctx, int32 pCount = 0 ) = 0;

   /** Just candy grammar for this->apply(vm); */
   void operator()( VMContext* ctx ) { invoke(ctx); }

   /** Return true if this function is ETA.
    \return true if the function is an ETA function.

    Eta functions are "functional constructs", that is, during functional
    evaluation, eta functions interrupt the normal sigma-reduction flow and
    are invoked to sigma-reduce their own parameters.
    */
   bool isEta() const { return m_bEta; }

   /** Set the Eta-ness status of this function.
    \param mode true to set this function as ETA.
    \see isEta()
    */
   void setEta( bool mode ) { m_bEta = mode; }

   inline Function* addParam( const String& name ) {
      m_params.insert( name );
      return this;
   }

   /** Candy grammar to declare parameters.
    In this way, it's possible to declare parameter of a function simply doing
    @code
    Function f;
    f << "po" << "p1" << "p2";
    @endcode

    To declare dynamic function in modules:

    @code
    Module *mod = new Module(...);
    (*mod)
       << &(*(new Func0) << "p0" << "p1" ... )
       << &(*(new Func1) << "p0" << "p1" ... << Function::eta );
    @endcode
    */
   inline Function& operator <<( const String& param )
   {
      addParam( param );
      return *this;
   }

   /** Setter for ETA function.
    \see setEta
    */
   class EtaSetter {
   };

   /** Setter for ETA function.

    Use this object to set the function as ETA in a compressed function declaration:
    @code
    Function f;
    f << "Param0" << "Param1" << Function::eta;
    @endcode
    */
   static EtaSetter eta;

   /** Candy grammar to set this function as eta.
    \see setEta
    */
   inline Function& operator <<( const EtaSetter& )
   {
      setEta(true);
      return *this;
   }
   
   /**
    * Returns the appropriate engine class handler for this Function.
    */
   virtual Class* handler() const;
   
   const SymbolMap& parameters() const  { return m_params; }
   SymbolMap& parameters() { return m_params; }

   const SymbolMap& locals() const  { return m_locals; }
   SymbolMap& locals() { return m_locals; }

   const SymbolMap& closed() const  { return m_closed; }
   SymbolMap& closed() { return m_closed; }

   inline uint32 paramCount() const { return m_params.size(); }
   bool hasClosure() const { return m_closed.size() != 0; }

   /**
    * Writes the function to an output stream.
    */
   virtual void render( TextWriter* tgt, int32 depth ) const;

   virtual void renderFunctionBody( TextWriter* tgt, int32 depth ) const;

   /** True if this function is used as main (global scope) for a module. */
   bool isMain() const { return m_bMain; }
   /** True if this function is used as main (global scope) for a module. */
   void setMain( bool m ) { m_bMain = m; }

   /** Return the module of this function, or of its class if this is a method. */
   Module* fullModule() const;

protected:
   SymbolMap m_params;
   SymbolMap m_locals;
   SymbolMap m_closed;
   String m_signature;
   Class* m_methodOf;
   bool m_bEta;
   bool m_bMain;
   
   Function() {}
   friend class ClassFunction;

};

#define FALCON_DECLARE_FUNCTION_EX(FN_NAME, SIGNATURE, extra ) \
   class Function_ ## FN_NAME: public ::Falcon::Function \
   { \
   public: \
      Function_ ## FN_NAME( ): \
         Function( #FN_NAME ) \
      { parseDescription( SIGNATURE ); } \
      virtual ~Function_ ## FN_NAME() {} \
      virtual void invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 pCount = 0 ); \
      extra\
   };


#define FALCON_DECLARE_FUNCTION(FN_NAME, SIGNATURE) \
         FALCON_DECLARE_FUNCTION_EX(FN_NAME, SIGNATURE, )

#define FALCON_DEFINE_FUNCTION(FN_NAME) void Function_ ## FN_NAME::invoke
#define FALCON_DEFINE_FUNCTION_P(FN_NAME) \
      void Function_ ## FN_NAME::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 pCount )
#define FALCON_DEFINE_FUNCTION_P1(FN_NAME) \
      void Function_ ## FN_NAME::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
#define FALCON_FUNCTION_NAME(FN_NAME) Function_ ## FN_NAME


#define FALCON_PCHECK( _pname , _pCheck ) ( _pname != 0 && _pname->is##_pCheck() )
#define FALCON_PCHECK_O( _pname , _pCheck ) ( _pname == 0 || _pname->isNil() || _pname->is##_pCheck() )

#define FALCON_PCHECK_GET( _pname , _pCheck, _tgt ) ( _pname != 0 && _pname->is##_pCheck() && ((_tgt= _pname->as##_pCheck()) || true) )
#define FALCON_PCHECK_O_GET( _pname , _pCheck, _tgt, _dflt ) ( \
        ((_pname == 0 || _pname->isNil()) && ((_tgt=_dflt) != 0|| true)) || \
        ( _pname->is##_pCheck() && ((_tgt=_pname->as##_pCheck()) != 0|| true) ) \
        )

#define FALCON_NPCHECK_GET( _pnum , _pCheck, _tgt ) FALCON_PCHECK_GET( (ctx->param(_pnum)), _pCheck, _tgt )
#define FALCON_NPCHECK_O_GET( _pnum , _pCheck, _tgt, _dflt ) FALCON_PCHECK_O_GET( (ctx->param(_pnum)), _pCheck, _tgt, _dflt )

}

#endif /* FUNCTION_H_ */

/* end of function.h */
