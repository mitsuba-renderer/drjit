How to make a new release?
--------------------------

1. Ensure that the most recent version of Dr.Jit is checked out (including all
   submodules).

2. Check that the ``nanobind`` dependency version in ``pyproject.toml`` (build
   requirement) matches the version used in the submodule.

3. Ensure that the changelog is up to date in ``docs/changelog.rst``. Run
   ``python3 resources/normalize_changelog_links.py`` to check that every
   commit link resolves.

4. Verify that the CI is currently green on all platforms.

5. Run the GHA "Build Python Wheels" with option "0". This effectively is a dry
   run of the wheel creation process.

6. If the action failed, fix whatever broke in the build process. If it
   succeeded, continue.

7. Update the version number in ``include/drjit/fwd.h``.

8. Add release number and date to ``docs/changelog.rst``.

9. Commit: ``git commit -am "vX.Y.Z release"``

10. Tag: ``git tag -a vX.Y.Z -m "vX.Y.Z release"``

11. Push: ``git push`` and ``git push --tags``

12. Run the GHA "Build Python Wheels" with option "1".

13. Check that the new version is available on
    `readthedocs <https://drjit.readthedocs.io/>`__.

14. Create the release on GitHub from the tag pushed at step 11:

    .. code-block:: bash

       python3 resources/changelog_to_release.py X.Y.Z --create

    This converts the matching section of ``docs/changelog.rst`` to Markdown
    and hands it to ``gh release create``. Omit ``--create`` to preview the
    notes, or add ``--draft`` to review them on GitHub before publishing.

15. Checkout the ``stable`` branch and run ``git pull --ff-only origin vX.Y.Z``
    and ``git push``
