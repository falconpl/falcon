#ifndef GTK_RECENTFILTER_HPP
#define GTK_RECENTFILTER_HPP

#include "modgtk.hpp"

#define GET_RECENTFILTER( item ) \
        ((GtkRecentFilter*)((Gtk::RecentFilter*) (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::RecentFilter
 */
class RecentFilter
    :
    public Gtk::CoreGObject
{
public:

    RecentFilter( const Falcon::CoreClass*, const GtkRecentFilter* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC set_name( VMARG );

    static FALCON_FUNC add_mime_type( VMARG );

    static FALCON_FUNC add_pattern( VMARG );

    static FALCON_FUNC add_pixbuf_formats( VMARG );

    static FALCON_FUNC add_application( VMARG );

    static FALCON_FUNC add_group( VMARG );

    static FALCON_FUNC add_age( VMARG );

    static FALCON_FUNC add_custom( VMARG );

    static gboolean exec_custom( const GtkRecentFilterInfo*, gpointer );

    static FALCON_FUNC get_needed( VMARG );

    static FALCON_FUNC filter( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_RECENTFILTER_HPP
