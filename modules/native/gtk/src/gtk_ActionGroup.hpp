/**
 *  \file gtk_ActionGroup.hpp
 */

#ifndef GTK_ACTIONGROUP_HPP
#define GTK_ACTIONGROUP_HPP

#include "modgtk.hpp"

#define GET_ACTIONGROUP( item ) \
        (((Gtk::ActionGroup*) (item).asObjectSafe() )->getObject())

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ActionGroup
 */
class ActionGroup
    :
    public Gtk::CoreGObject
{
public:

    ActionGroup( const Falcon::CoreClass*, const GtkActionGroup* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkActionGroup* getObject() const { return (GtkActionGroup*) m_obj; }

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC add_action_with_accel( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_ACTIONGROUP_HPP

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
