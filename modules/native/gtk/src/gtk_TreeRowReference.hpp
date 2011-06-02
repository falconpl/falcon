#ifndef GTK_TREEROWREFERENCE_HPP
#define GTK_TREEROWREFERENCE_HPP

#include "modgtk.hpp"

#define GET_TREEROWREFERENCE( item ) \
        (((Gtk::TreeRowReference*) (item).asObjectSafe() )->getTreeRowReference())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeRowReference
 */
class TreeRowReference
    :
    public Falcon::CoreObject
{
public:

    TreeRowReference( const Falcon::CoreClass*,
              const GtkTreeRowReference* = 0, const bool transfer = false );

    ~TreeRowReference();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkTreeRowReference* getTreeRowReference() const { return (GtkTreeRowReference*) m_rowref; }

    void setTreeRowReference( const GtkTreeRowReference*, const bool transfer = false );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_proxy( VMARG );

    static FALCON_FUNC get_model( VMARG );

    static FALCON_FUNC get_path( VMARG );

    static FALCON_FUNC valid( VMARG );

#if 0 // unused
    static FALCON_FUNC free( VMARG );
#endif

    static FALCON_FUNC copy( VMARG );

    static FALCON_FUNC inserted( VMARG );

    static FALCON_FUNC deleted( VMARG );

    static FALCON_FUNC reordered( VMARG );

private:

    GtkTreeRowReference*    m_rowref;

};


} // Gtk
} // Falcon

#endif // !GTK_TREEROWREFERENCE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
