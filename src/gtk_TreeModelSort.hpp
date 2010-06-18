#ifndef GTK_TREEMODELSORT_HPP
#define GTK_TREEMODELSORT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeModelSort
 */
class TreeModelSort
    :
    public Gtk::CoreGObject
{
public:

    TreeModelSort( const Falcon::CoreClass*, const GtkTreeModelSort* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC get_model( VMARG );

    static FALCON_FUNC convert_child_path_to_path( VMARG );

    static FALCON_FUNC convert_child_iter_to_iter( VMARG );

    static FALCON_FUNC convert_path_to_child_path( VMARG );

    static FALCON_FUNC convert_iter_to_child_iter( VMARG );

    static FALCON_FUNC reset_default_sort_func( VMARG );

    static FALCON_FUNC clear_cache( VMARG );

    static FALCON_FUNC iter_is_valid( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TREEMODELSORT_HPP
