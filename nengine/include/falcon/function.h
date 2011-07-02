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

   /** Returns the name of this function. */
   const String& name() const { return m_name; }

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

   /** Mark this function for garbage collecting. */
   void gcMark( int32 mark );

   /** Store in a garbage collector. 
    \param c A collector where to store this function.
    
    When this method is called, the function become subject to garbage
    collection.
    */
   GCToken* garbage( Collector* c );

   Error* paramError( int line = 0, const char* modName = 0 ) const;

   /** Garbage this function on the standard collector. */
   GCToken* garbage();

   /** Executes the call.
    \param ctx The Virtual Machine context on which the function is executed.
    \param pCount Number of parameters in the stack for this function.
    
    The call execution may be either immediate or deferred; for example,
    the call may just leaves PSteps to be executed by the virtual machine.

    In case of deferred calls, apply must also push proper return PStep codes.
    In case of immediate calls, apply() must also perform the return frame
    code in the virtual machine by calling VMachine::returnFrame().
    */
   virtual void apply( VMContext* ctx, int32 pCount = 0 ) = 0;

   /** Just candy grammar for this->apply(vm); */
   void operator()( VMContext* ctx ) { apply(ctx); }

   /** Return true if this function is deterministic.
    \return true if the function is deterministic.
    */
   bool isDeterm() const { return m_bDeterm; }
   
   /** Set the determinism status of this function.
    \param mode true to set this function as deterministic
    */
   void setDeterm( bool mode ) { m_bDeterm = mode; }

   /** Return true if this function is ETA.
    \return true if the function is an ETA function.

    Eta functions are "functional constructs", that is, during functional
    evaluation, eta functions interrupt the normal sigma-reduction flow and
    are invoked to sigma-reduce their own parameters.
    */
   bool isEta() const { return m_bEta; }

   /** Set the determinism status of this function.
    \param mode true to set this function as deterministic
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
       << &(*(new Func1) << "p0" << "p1" ... << Function::determ );
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

   /** Deterministic setter.
    \see setDeterm
    */
   class DetermSetter {
   };

   /** Deterministic setter.
    Use this object to set the function as deterministic:
    @code
    Function f;
    f << "Param0" << "Param1" << Function::determ;
    @endcode
    */
   static DetermSetter determ;

   /** Candy grammar to set this function as eta.
    \see setEta
    */
   inline Function& operator <<( const EtaSetter& )
   {
      setEta(true);
      return *this;
   }

   /** Candy grammar to set this function as deterministic.
    \see setDeterm
    */
   inline Function& operator <<( const DetermSetter& )
   {
      setDeterm(true);
      return *this;
   }

protected:
   String m_name;
   String m_signature;
   
   int32 m_paramCount;

   GCToken* m_gcToken;   
   Module* m_module;

   int32 m_line;

   bool m_bDeterm;
   bool m_bEta;

   SymbolTable m_symtab;
};

}

#endif /* FUNCTION_H_ */

/* end of function.h */
