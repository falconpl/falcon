#ifndef GTK_TREEPATH_HPP
#define GTK_TREEPATH_HPP

#include "modgtk.hpp"

#define GET_TREEPATH( item ) \
        (((Gtk::TreePath*) (item).asObjectSafe() )->getTreePath())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreePath
 */
class TreePath
    :
    public Falcon::CoreObject
{
public:

    TreePath( const Falcon::CoreClass*,
              const GtkTreePath* = 0, const bool transfer = false );

    ~TreePath();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkTreePath* getTreePath() const { return (GtkTreePath*) m_path; }

    void setTreePath( const GtkTreePath*, const bool transfer = false );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_from_string( VMARG );

#if 0 // unused
    static FALCON_FUNC new_from_indices( VMARG );
#endif

    static FALCON_FUNC to_string( VMARG );

    static FALCON_FUNC new_first( VMARG );

    static FALCON_FUNC append_index( VMARG );

    static FALCON_FUNC prepend_index( VMARG );

    static FALCON_FUNC get_depth( VMARG );

    static FALCON_FUNC get_indices( VMARG );

#if 0 // unused, see get_indices (arr len = depth)
    static FALCON_FUNC get_indices_with_depth( VMARG );
#endif

#if 0 // unused
    static FALCON_FUNC free( VMARG );
#endif

    static FALCON_FUNC copy( VMARG );

    static FALCON_FUNC compare( VMARG );

    static FALCON_FUNC next( VMARG );

    static FALCON_FUNC prev( VMARG );

    static FALCON_FUNC up( VMARG );

    static FALCON_FUNC down( VMARG );

    static FALCON_FUNC is_ancestor( VMARG );

    static FALCON_FUNC is_descendant( VMARG );

private:

    GtkTreePath*    m_path;

};


} // Gtk
} // Falcon

#endif // !GTK_TREEPATH_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
