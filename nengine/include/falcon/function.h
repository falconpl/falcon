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
#include <falcon/symboltable.h>

namespace Falcon
{

class Collector;
class VMContext;
class Error;
class Class;

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

class FALCON_DYN_CLASS Function
{
public:
   Function( const String& name, Module* owner = 0, int32 line = 0 );
   virtual ~Function();
   
   /** Sets the module of this function.
    Mainly, this information is used for debugging (i.e. to know where a function
    is declared).
    */
   void module( Module* owner );

   /** Return the module where this function is allocated.
   */
   Module* module() const { return m_module; }

   /** Removes the link between a function and a module.
    Used by the module destructor do prevent de-referencing during destruction.
   */
   void detachModule() {m_module = 0;}
   
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
    parseDescription( "param0:S,param1:[N]" );
    \endcode
    
    are equivalent.
    
    The description may start with "&" character. "&" means that
    the function is ETA (by default, functions are non-eta).
    
    \return false on malformed parameter string, 
    */
   bool parseDescription( const String& desc );
   
   
   //void getParams( int pCount,  ... );

   void methodOf( Class* cls ) { m_methodOf = cls; }
   Class* methodOf() const { return m_methodOf; }


   /** Returns the name of this function. */
   const String& name() const { return m_name; }

   /** Renames the function.
    \param n The new name of the function.
    \note Will throw an assertion if already stored in a module.
    */
   void name( const String& n ) { m_name = n; }

   /** Sets the signature of the function.
    \param sign A string with the expected parameters of the function.
    */
   void signature( const String& sign ) { m_signature = sign; }

   /** Gets the signature of the function.
    \return A string representing the expected parameters of the function.
    */
   const String& signature() const { return m_signature; }

   /** Returns the source line where this function was declared.
    To be used in conjunction with module() to pinpoint the location of a function.
    */
   int32 declaredAt() const { return m_line; }

   /** Returns the complete identifier for this function.

    The format is:
         name(line) source/path.fal

    Where a source path is not available, the module full qualified name is used.
   
   */
   String locate() const;   

   int32 paramCount() const { return m_paramCount; }

   /** Sets the variable count.
    * @param pc The number of parameters in this functions.
    *
    * the parameter count should be a number in 0..varCount().
    */
   void paramCount( int32 pc ) { m_paramCount = pc; }

   Error* paramError( int line = 0, const char* modName = 0 ) const;

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


   /** Returns the symbol table of this function.
    */
   const SymbolTable& symbols() const { return m_symtab; }

   /** Returns the symbol table of this function.
    */
   SymbolTable& symbols() { return m_symtab; }

   /** Adds a parameter to this function.
    \param param The parameter to be added.
    \note call this before accessing symbols() in this function.
    */
   inline void addParam( const String& param )
   {
      m_symtab.addLocal( param );
      m_paramCount = m_symtab.localCount();
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
      m_symtab.addLocal( param );
      m_paramCount = m_symtab.localCount();
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

   /** GCMark this function.
      
    Virtual because some function subclasses having closed items
      may need to mark their own items.

    The base class version just saves the mark for gcCheck().
    */
   virtual void gcMark( uint32 mark );

   /** GCMark this function.

    Virtual because some function subclasses having closed items
      may need to mark their own items.

    The base class destroys itself and return false if the mark is
    newer than the one seen in gcMark.
    */
   virtual bool gcCheck( uint32 mark );

protected:
   String m_name;
   String m_signature;
   
   int32 m_paramCount;

   uint32 m_lastGCMark;
   
   Module* m_module;
   Class* m_methodOf;

   int32 m_line;

   bool m_bEta;
   SymbolTable m_symtab;
};

}

#endif /* FUNCTION_H_ */

/* end of function.h */
