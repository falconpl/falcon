/*
   FALCON - The Falcon Programming Language.
   FILE: modrequest.h

   Structure recording the information about foreign module load.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 12 Feb 2012 14:13:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MODREQUEST_H_
#define _FALCON_MODREQUEST_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {

class Module;

/** Class keeping records of modules requested by a module. 
 This class is used by modules to keep track of the other modules they need
 to load or find.
 
 The structure also carries the modules once they are found; it doesn't own
 nor reference them; they are GC Marked by the host module on need.
 
 Notice that it is possible that multiple ModRequest filed under the same module
 can refer to the same target module. This happens if multiple names initially
 thought to be different actually resolve into the same phisical module entity 
 during the load step. For instance, a module loaded by a relative URI can
 be found to be the same module as another one invoked by absolute URI; their
 phisical name differs until, at runtime, it is possible to determine their
 real complete URI. 
 Again, the engine may override load requests and provide the same module for for
 different module names. 
 
 This is another reason why Module entities are never
 owned by ModRequest.
 */
class ModRequest
{
public:
   ModRequest();
   ModRequest( const String& name, bool isUri = false, bool isLoad = false, Module* mod = 0);   
   ModRequest( const ModRequest& other );   
   ~ModRequest();

   inline Module* module() const { return m_module; }
   inline const String& name() const { return m_name; }   
   inline bool isLoad() const { return m_isLoad; }
   inline bool isUri() const { return m_bIsURI; }
   
   inline void isUri(bool v) { m_bIsURI = v ; }
   
   inline void promoteLoad() { m_isLoad = true; }
   inline void resolved( Module* mod ) { m_module = mod; }
      
   /** Position of this entity in the module data.
   Used for simpler serialization, so that it is possible to reference this
   entity in the serialized file.
   */
   inline int id() const { return m_id; }
   
   /** Sets the position of this entity in the module data.
   Used for simpler serialization, so that it is possible to reference this
   entity in the serialized file.
   */
   inline void id( int n ) { m_id = n; }
   
   
   void addImportDef( ImportDef* id );
   void removeImportDef( ImportDef* id );
   int importDefCount() const;
   
private:
   class ImportDefList;
   ImportDefList* m_idl;
   
   String m_name;
   bool m_isLoad;
   bool m_bIsURI;
   Module* m_module;
  
   int m_id;
};

}

#endif	/* MODREQUEST_H */

/* end of modrequest.h */
