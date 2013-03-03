/**
 *  \file gtk_AccelGroup.hpp
 */

#ifndef GTK_ACCELGROUP_HPP
#define GTK_ACCELGROUP_HPP

#include "modgtk.hpp"

#define GET_ACCELGROUP( item ) \
        (((Gtk::AccelGroup*) (item).asObjectSafe() )->getObject())

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::AccelGroup
 */
class AccelGroup
    :
    public Gtk::CoreGObject
{
public:

    AccelGroup( const Falcon::CoreClass*, const GtkAccelGroup* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkAccelGroup* getObject() const { return (GtkAccelGroup*) m_obj; }

    static FALCON_FUNC init( VMARG );

    static gboolean activate_cb( GtkAccelGroup*, GObject*, guint, GdkModifierType, gpointer );

    static FALCON_FUNC connect_group( VMARG ); // renamed to avoid conflict with GObject.connect()
#if 0
    static FALCON_FUNC connect_by_path( VMARG );

    static FALCON_FUNC disconnect( VMARG );

    static FALCON_FUNC disconnect_key( VMARG );

    static FALCON_FUNC query( VMARG );

    static FALCON_FUNC activate( VMARG );

    static FALCON_FUNC lock( VMARG );

    static FALCON_FUNC unlock( VMARG );

    static FALCON_FUNC get_is_locked( VMARG );

    static FALCON_FUNC from_accel_closure( VMARG );

    static FALCON_FUNC get_modifier_mask( VMARG );

#if 0
gboolean            gtk_accel_groups_activate           (GObject *object,
                                                         guint accel_key,
                                                         GdkModifierType accel_mods);
GSList *            gtk_accel_groups_from_object        (GObject *object);
GtkAccelKey *       gtk_accel_group_find                (GtkAccelGroup *accel_group,
                                                         GtkAccelGroupFindFunc find_func,
                                                         gpointer data);
#endif
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_ACCELGROUP_HPP

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
