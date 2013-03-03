/*
   FALCON - The Falcon Programming Language.
   FILE: vardatamap.h

   Map holding variables and associated data for global storage.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 08 Jan 2013 18:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_VARDATAMAP_H_
#define _FALCON_VARDATAMAP_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>
#include <falcon/variable.h>

namespace Falcon {

class String;
class Module;

/** Map holding variables and associated data for global storage in modules.
 *
 */
class FALCON_DYN_CLASS VarDataMap
{
public:
   VarDataMap();
   ~VarDataMap();

   class VarData
   {
   public:
      VarData( const String& n, Variable v, const Item& data, bool bExtern = false, bool bExported = false ):
         m_name(n),
         m_var(v),
         m_storage( data ),
         m_bExported( bExported )
      {
         if( ! bExtern ) {
            m_data = &m_storage;
         }
      }

      String m_name;
      Variable m_var;
      Item* m_data;
      Item m_storage;
      bool m_bExported;
   };


   /** Adds a global variable, possibly exportable.
      \param name The name of the symbol referencing the variable.
      \param bExport if true, the returned symbol will be exported.
      \param value the value to be added.
      \return A variable containing the global ID or 0 if the variable
      was already declared.

      Creates a variable with a value in the module global vector.

      \note The method won't check the contents of the item; if it's a function or a class,
      it will normally be available as a global module variable, but it won't be added
      to the module mantra map.

      */
   VarData* addGlobal( const String& name, const Item& value, bool bExport = false );

   VarData* addExtern( const String& name, bool bExport = false );

   bool removeGlobal( const String& name );
   bool removeGlobal( uint32 id );

  /** Export a symbol.
   \param name The name of the symbol to be exported.
   \param bAlready will be set to true if the symbol was already expored.
   \return The exported variable or zero if the name isn't known.

   \note In the falcon language it is NOT legal to export undefined symbols -- to avoid mistyping.
   */
   VarData* addExport( const String& name, bool &bAlready );

  inline VarData* addExport( const String& name ) {
     bool dummy;
     return addExport(name, dummy);
  }

   /** Finds the value of a global variable by name.
   \param name The variable name to be searched.
   \return A pointer to a global value, or 0 if not found.

   Notice that the returned value might be either defined as a static
   value in this module or being imported from other sources (i.e. the
   exported data in the host module space).

   \note Still unimported extern variables return 0.
   */
   Item* getGlobalValue( const String& name ) const;

   /** Finds the value of a global variable by its variable id.
   \param name The variable name to be searched.
   \return A global value.

   Notice that the returned value might be either defined as a static
   value in this module or being imported from other sources (i.e. the
   exported data in the host module space).

   \note Still unimported extern variables return 0.
   */
   Item* getGlobalValue( uint32 id ) const;

   /** Finds the global variable associated with a given variable name.
       \param name The variable name to be searched.
       \return The variable definition, if found, or 0.
   */
   VarData* getGlobal( const String& name ) const;

   /** Finds a variable definition and name given its id.
       \param id The global ID of the desired variable.
       \param name The variable name of the required global.
       \param var The variable definition of the required global.
       \return true if id is in range, false otherwise.

       Notice that there isn't any distinction between global variables
       defined by this module and extern variables imported here.

       The value of the required variable can be directly accessed by invoking
       getGlobalValue().
   */
   VarData* getGlobal( uint32 id ) const;

   /** Checks if a variable is exported given its name.
   */
   bool isExported( const String& name ) const;

   /** Mark (dynamic) modules for Garbage Collecting.
    \param mark the current GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */
   void gcMark( uint32 mark );

   /** Determines if a module can be reclaimed.
    \return last GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */

   uint32 lastGCMark() const { return m_lastGCMark; }


   class VariableEnumerator
   {
   public:
      virtual void operator() ( const String& name, const Variable& var, const Item& value ) = 0;
   };


   /** Enumerate all the variables exported by this module.
    *
    * By definition, exported variables are global and provide a valid pointer
    * when getGlobalValue() is invoked.
    */
   void enumerateExports( VariableEnumerator& rator ) const;

   /** Returns the count of global variables. */
   uint32 size() const;

   /** Promotes a variable previously known as extern into a global.
    * \param id The variable to be promoted (known by ID).
    * \param value The concrete value that is given.
    * \param redeclaredAt New line where the promotion is done.
    * \return true if the variable was an extern, false otherwise.
    *
    * This turns an extern variable into a global, eventually removing the
    * extern dependencies bound with the variable name.
    *
    * \note; on exit, the \b ext variable is external.
    */
   bool promoteExtern( uint32 id, const Item& value, int32 redeclaredAt=0 );

   void forwardNS( VarDataMap* other, const String& remoteNS, const String& localNS );
   void exportNS( Module* source, const String& sourceNS, Module* target, const String& targetNS );

   void store( DataWriter* dw ) const;
   void restore( DataReader* dr );

   void flatten( VMContext* ctx, ItemArray& subItems ) const;
   void unflatten( VMContext* ctx, ItemArray& subItems, uint32 start, uint32 &count );

private:
   class Private;
   VarDataMap::Private* _p;

   Variable* addGlobalVariable( const String& name, Item* pvalue );

   uint32 m_lastGCMark;
};

}

#endif

/* end of vardatamap.h */
