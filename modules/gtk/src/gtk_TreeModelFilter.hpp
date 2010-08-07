#ifndef GTK_TREEMODELFILTER_HPP
#define GTK_TREEMODELFILTER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeModelFilter
 */
class TreeModelFilter
    :
    public Gtk::CoreGObject
{
public:

    TreeModelFilter( const Falcon::CoreClass*, const GtkTreeModelFilter* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_visible_func( VMARG );

    static gboolean exec_visible_func( GtkTreeModel*, GtkTreeIter*, gpointer );
#if 0 // todo
    static FALCON_FUNC set_modify_func( VMARG );
#endif
    static FALCON_FUNC set_visible_column( VMARG );

    static FALCON_FUNC get_model( VMARG );

    static FALCON_FUNC convert_child_iter_to_iter( VMARG );

    static FALCON_FUNC convert_iter_to_child_iter( VMARG );

    static FALCON_FUNC convert_child_path_to_path( VMARG );

    static FALCON_FUNC convert_path_to_child_path( VMARG );

    static FALCON_FUNC refilter( VMARG );

    static FALCON_FUNC clear_cache( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TREEMODELFILTER_HPP
